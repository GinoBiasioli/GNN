import argparse
import json
import logging
import os
import random
import traceback
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ax.service.managed_loop import optimize
from ax.service.utils.report_utils import exp_to_df
from torch.nn import Linear, Module, ModuleDict, ModuleList
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, HeteroConv, global_mean_pool
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from tqdm import tqdm


DEFAULT_DATASET = "bpi_2012"
DEFAULT_GRAPH_FOLDER_NAME = "object_hetero_graphs"
TARGET_NODE_TYPE = "event"
PATIENCE = 10
NUM_EPOCHS = 50
DEFAULT_TRIALS = 10
DEFAULT_RUNS = 10


def resolve_project_root() -> str:
    env_root = os.environ.get("THESIS_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "data" / "dataset_features.json").exists():
            return str(candidate)
        raise FileNotFoundError(
            "THESIS_PROJECT_ROOT is set, but data/dataset_features.json "
            f"was not found under: {candidate}"
        )

    candidates = []
    if "__file__" in globals():
        script_path = Path(__file__).resolve()
        candidates.extend([script_path.parent, script_path.parent.parent])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, cwd / "Thesis code", cwd.parent])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "data" / "dataset_features.json").exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not locate the thesis project root. "
        "Expected to find data/dataset_features.json."
    )


def resolve_dir(env_var_name: str, default_path: str) -> str:
    env_value = os.environ.get(env_var_name)
    if env_value:
        return os.path.abspath(os.path.expanduser(env_value))
    return default_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a heterogeneous GNN on object-centric prefix graphs."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=os.environ.get("THESIS_DATASET", DEFAULT_DATASET),
        help="Dataset folder under data/datasets/object_hetero_graphs.",
    )
    parser.add_argument(
        "--prediction-task",
        choices=["auto", "next_activity", "next_event"],
        default="auto",
        help=(
            "next_activity uses y_activity only. next_event uses y_activity, "
            "y_resource, and y_time when those labels are present."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Maximum training epochs per run.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of Ax hyperparameter-search trials.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of final repeated test runs.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the smoke-test configuration and skip Ax/final repeated runs.",
    )
    return parser.parse_args()


def set_random_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_dataset_file(graph_dir: str, dataset: str, file_name: str) -> tuple[list[HeteroData], dict]:
    path = os.path.join(graph_dir, dataset, file_name)
    size = os.path.getsize(path) / (1024**3)
    print(f'\nImporting "{file_name}" ({size:.2f} GB)')
    loaded_data = torch.load(path, weights_only=False)
    summary = {
        "dataset": dataset,
        "file_name": file_name,
        "graphs_count": len(loaded_data),
        "file_size_gb": round(size, 4),
    }
    print(f'Imported "{file_name}": {summary["graphs_count"]} graphs')
    return loaded_data, summary


def load_object_graph_splits(graph_dir: str, dataset: str) -> tuple[list[HeteroData], list[HeteroData], list[HeteroData], list[dict]]:
    dataset_import_summaries = []
    split_files = [
        "train_set_object_hetero.pt",
        "validation_set_object_hetero.pt",
        "test_set_object_hetero.pt",
    ]

    loaded = {}
    for file_name in tqdm(split_files):
        data, summary = load_dataset_file(graph_dir, dataset, file_name)
        loaded[file_name] = data
        dataset_import_summaries.append(summary)

    return (
        loaded["train_set_object_hetero.pt"],
        loaded["validation_set_object_hetero.pt"],
        loaded["test_set_object_hetero.pt"],
        dataset_import_summaries,
    )


def collect_metadata(graphs: list[HeteroData]) -> tuple[list[str], list[tuple[str, str, str]]]:
    node_types = set()
    edge_types = set()
    for graph in graphs:
        graph_node_types, graph_edge_types = graph.metadata()
        node_types.update(graph_node_types)
        edge_types.update(graph_edge_types)
    return sorted(node_types), sorted(edge_types)


def has_label(graph: HeteroData, attr_name: str) -> bool:
    return hasattr(graph, attr_name) and getattr(graph, attr_name) is not None


def resolve_prediction_task(
    requested_task: str,
    sample_graph: HeteroData,
) -> str:
    if requested_task != "auto":
        return requested_task

    if has_label(sample_graph, "y_resource") and has_label(sample_graph, "y_time"):
        return "next_event"

    return "next_activity"


def validate_prediction_labels(graphs: list[HeteroData], prediction_task: str) -> None:
    required_labels = ["y_activity"]
    if prediction_task == "next_event":
        required_labels.extend(["y_resource", "y_time"])

    missing = [
        label
        for label in required_labels
        if not all(has_label(graph, label) for graph in graphs)
    ]
    if missing:
        raise ValueError(
            f"Prediction task '{prediction_task}' requires labels {missing}, "
            "but at least one graph does not contain them. Regenerate the object "
            "graphs with those targets or use --prediction-task next_activity."
        )


def infer_output_spaces(
    train_graphs: list[HeteroData],
    valid_graphs: list[HeteroData],
    test_graphs: list[HeteroData],
    prediction_task: str,
) -> tuple[dict[str, int], dict[str, int]]:
    all_graphs = train_graphs + valid_graphs + test_graphs
    validate_prediction_labels(all_graphs, prediction_task)

    activity_labels = [int(graph.y_activity.item()) for graph in all_graphs]
    output_cat = {"Activity": max(activity_labels) + 1}
    output_real = {}

    if prediction_task == "next_event":
        resource_labels = [int(graph.y_resource.item()) for graph in all_graphs]
        output_cat["org:resource"] = max(resource_labels) + 1
        output_real["time:timestamp"] = 1

    return output_cat, output_real


def get_labels(batch: HeteroData, output_cat: dict[str, int], output_real: dict[str, int]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    labels_cat = {}
    labels_real = {}

    if "Activity" in output_cat:
        labels_cat["Activity"] = batch.y_activity.view(-1).long()
    if "org:resource" in output_cat:
        labels_cat["org:resource"] = batch.y_resource.view(-1).long()
    if "time:timestamp" in output_real:
        labels_real["time:timestamp"] = batch.y_time.view(-1, 1).float()

    return labels_cat, labels_real


class ObjectHeteroGNN(Module):
    def __init__(
        self,
        output_cat: dict[str, int],
        output_real: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        parameters: dict,
        target_node_type: str = TARGET_NODE_TYPE,
    ) -> None:
        super().__init__()

        hid = parameters["hid"]
        layers = parameters["layers"]
        aggregation = parameters["aggregation"]

        self.output_cat = output_cat
        self.output_real = output_real
        self.target_node_type = target_node_type
        self.convs = ModuleList()

        for _ in range(layers):
            conv = HeteroConv(
                {
                    relation: GATv2Conv(
                        (-1, -1),
                        hid,
                        heads=1,
                        concat=False,
                        add_self_loops=False,
                        residual=False,
                    )
                    for relation in edge_types
                },
                aggr=aggregation,
            )
            self.convs.append(conv)

        self.fc_cat = ModuleDict(
            {name: Linear(hid, output_cat[name]) for name in output_cat}
        )
        self.fc_real = ModuleDict(
            {name: Linear(hid, 1) for name in output_real}
        )

    def forward(self, batch: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = batch.x_dict

        for conv in self.convs:
            x_dict = conv(x_dict, batch.edge_index_dict)
            x_dict = {node_type: F.relu(x) for node_type, x in x_dict.items()}

        pooled = global_mean_pool(
            x_dict[self.target_node_type],
            batch[self.target_node_type].batch,
        )

        output = {}
        for name, head in self.fc_cat.items():
            output[name] = head(pooled)
        for name, head in self.fc_real.items():
            output[name] = head(pooled)

        return output


def evaluate_model(
    net: ObjectHeteroGNN,
    loader: DataLoader,
    output_cat: dict[str, int],
    output_real: dict[str, int],
    device: torch.device,
) -> dict:
    criterion_cat = nn.CrossEntropyLoss()
    criterion_real = nn.L1Loss()

    running_total_loss = []
    running_cat_loss = {name: [] for name in output_cat}
    running_real_loss = {name: [] for name in output_real}
    predictions_categorical = {name: [] for name in output_cat}
    target_categorical = {name: [] for name in output_cat}
    prediction_numerical = {name: [] for name in output_real}
    target_numerical = {name: [] for name in output_real}
    avg_mae = {name: [] for name in output_real}

    net.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels_cat, labels_real = get_labels(batch, output_cat, output_real)
            outputs = net(batch)

            batch_loss = torch.tensor(0.0, device=device)
            for name in output_cat:
                current_loss = criterion_cat(outputs[name], labels_cat[name])
                batch_loss = batch_loss + current_loss
                running_cat_loss[name].append(current_loss.detach())
                preds = torch.argmax(torch.softmax(outputs[name], dim=1), dim=1)
                predictions_categorical[name].append(preds.detach())
                target_categorical[name].append(labels_cat[name].detach())

            for name in output_real:
                current_loss = criterion_real(outputs[name], labels_real[name])
                batch_loss = batch_loss + current_loss
                running_real_loss[name].append(current_loss.detach())
                avg_mae[name].append(
                    torch.mean(torch.abs(outputs[name] - labels_real[name])).detach()
                )
                prediction_numerical[name].append(outputs[name].detach())
                target_numerical[name].append(labels_real[name].detach())

            running_total_loss.append(batch_loss.detach())

    total_loss = torch.stack(running_total_loss).mean()

    avg_cat_loss = {
        name: torch.stack(running_cat_loss[name]).mean()
        for name in output_cat
    }
    avg_real_loss = {
        name: torch.stack(running_real_loss[name]).mean()
        for name in output_real
    }

    macro_f1s = {}
    accuracy = {}
    for name in output_cat:
        predictions_categorical[name] = torch.cat(predictions_categorical[name])
        target_categorical[name] = torch.cat(target_categorical[name])
        macro_f1s[name] = multiclass_f1_score(
            predictions_categorical[name],
            target_categorical[name],
            num_classes=output_cat[name],
            average="macro",
        )
        accuracy[name] = multiclass_accuracy(
            predictions_categorical[name],
            target_categorical[name],
            num_classes=output_cat[name],
        )

    for name in output_real:
        prediction_numerical[name] = torch.cat(prediction_numerical[name]).view(-1)
        target_numerical[name] = torch.cat(target_numerical[name]).view(-1)
        avg_mae[name] = torch.stack(avg_mae[name]).mean()

    return {
        "loss": total_loss,
        "details": {
            "loss": total_loss.item(),
            **{f"{name}_loss": avg_cat_loss[name].item() for name in avg_cat_loss},
            **{f"{name}_loss": avg_real_loss[name].item() for name in avg_real_loss},
            **{f"{name}_acc": accuracy[name].item() for name in accuracy},
            **{f"{name}_macroF1": macro_f1s[name].item() for name in macro_f1s},
            **{f"{name}_MAE": avg_mae[name].item() for name in avg_mae},
        },
    }


def train_object_hgnn(
    config: dict,
    train_graphs: list[HeteroData],
    valid_graphs: list[HeteroData],
    output_cat: dict[str, int],
    output_real: dict[str, int],
    edge_types: list[tuple[str, str, str]],
    device: torch.device,
    num_epochs: int,
) -> tuple[dict, ObjectHeteroGNN]:
    print(config)

    net = ObjectHeteroGNN(
        parameters=config,
        output_cat=output_cat,
        output_real=output_real,
        edge_types=edge_types,
    ).to(device)

    criterion_cat = nn.CrossEntropyLoss()
    criterion_real = nn.L1Loss()

    train_loader = DataLoader(
        train_graphs,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_graphs,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(PATIENCE / 2),
    )

    best_model = None
    best_loss = None
    best_eval_details = None
    best_f1 = 0
    pat_count = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for epoch in tqdm(range(num_epochs)):
        net.train()
        for batch in train_loader:
            batch = batch.to(device)
            labels_cat, labels_real = get_labels(batch, output_cat, output_real)

            optimizer.zero_grad()
            outputs = net(batch)

            total_loss = torch.tensor(0.0, device=device)
            for name in output_cat:
                total_loss = total_loss + criterion_cat(outputs[name], labels_cat[name])
            for name in output_real:
                total_loss = total_loss + criterion_real(outputs[name], labels_real[name])

            total_loss.backward()
            optimizer.step()

        eval_result = evaluate_model(net, valid_loader, output_cat, output_real, device)
        val_loss = eval_result["loss"]
        eval_details = {
            f"valid_{key}": value
            for key, value in eval_result["details"].items()
        }
        lr_scheduler.step(val_loss)

        f1_activity = eval_details.get("valid_Activity_macroF1", 0)
        if best_loss is None or val_loss < best_loss:
            best_model = deepcopy(net)
            best_loss = val_loss
            best_eval_details = eval_details.copy()
            pat_count = 0
        else:
            pat_count += 1

        if best_f1 < f1_activity:
            best_f1 = f1_activity

        print("/" * 10)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(eval_details)
        print(
            f"Patience {pat_count}/{PATIENCE}, "
            f"val loss {val_loss.item()} "
            f"current_lr {lr_scheduler.get_last_lr()}, "
            f"curr_best_activity_F1 {best_f1}"
        )

        if pat_count == PATIENCE:
            return best_eval_details, best_model

    return best_eval_details, best_model


def test_object_hgnn(
    net: ObjectHeteroGNN,
    test_graphs: list[HeteroData],
    output_cat: dict[str, int],
    output_real: dict[str, int],
    device: torch.device,
) -> dict:
    test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)
    eval_result = evaluate_model(net, test_loader, output_cat, output_real, device)
    details = eval_result["details"]
    result = {
        **{key: value for key, value in details.items() if key != "loss"},
        "AVG_total_loss": details["loss"],
    }
    print(result)
    return result


def create_df(results: list[dict]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    result_table = pd.DataFrame(results)
    return result_table, result_table.mean(numeric_only=True), result_table.std(numeric_only=True)


def test_multi(
    config: dict,
    train_graphs: list[HeteroData],
    valid_graphs: list[HeteroData],
    test_graphs: list[HeteroData],
    output_cat: dict[str, int],
    output_real: dict[str, int],
    edge_types: list[tuple[str, str, str]],
    device: torch.device,
    num_epochs: int,
    num_runs: int,
    save_path: str,
) -> list[dict]:
    results = []
    os.makedirs(save_path, exist_ok=True)

    for run_idx in range(num_runs):
        print(f"Run {run_idx}")
        _, net = train_object_hgnn(
            config,
            train_graphs,
            valid_graphs,
            output_cat,
            output_real,
            edge_types,
            device,
            num_epochs,
        )
        results.append(
            test_object_hgnn(
                net,
                test_graphs,
                output_cat,
                output_real,
                device,
            )
        )
        print("RES:")
        print(results[-1])

    results_table, means, stds = create_df(results)
    results_table.to_csv(os.path.join(save_path, "results_object_hetero.csv"), index=False)
    pd.DataFrame({"mean": means, "std": stds}).to_csv(
        os.path.join(save_path, "mean_and_stds_object_hetero.csv")
    )
    print(pd.DataFrame({"mean": means, "std": stds}))
    print(f"Saved final object-hetero evaluation results to: {save_path}")
    return results


def main() -> None:
    args = parse_args()
    global NUM_EPOCHS
    NUM_EPOCHS = args.epochs

    logging.getLogger("root").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.serialization.add_safe_globals([HeteroData])

    root_path = resolve_project_root()
    graph_dir = resolve_dir(
        "THESIS_OBJECT_GRAPHS_DIR",
        os.path.join(root_path, "data", "datasets", DEFAULT_GRAPH_FOLDER_NAME),
    )
    results_root_dir = resolve_dir(
        "THESIS_RESULTS_DIR",
        os.path.join(root_path, "results"),
    )
    results_dir = os.path.join(results_root_dir, args.dataset, "object_hetero")
    os.makedirs(results_dir, exist_ok=True)

    pd.set_option("display.max_columns", None)
    set_random_seeds(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(root_path, graph_dir, results_dir, sep="\n")
    print(f"Selected dataset: {args.dataset}")
    print(f"Using device: {device}")

    x_train, x_valid, x_test, import_summaries = load_object_graph_splits(
        graph_dir,
        args.dataset,
    )
    pd.DataFrame(import_summaries).to_csv(
        os.path.join(results_dir, "dataset_import_summary_object_hetero.csv"),
        index=False,
    )

    node_types, edge_types = collect_metadata(x_train + x_valid + x_test)
    print("Node types:", node_types)
    print("Edge types:", edge_types)

    prediction_task = resolve_prediction_task(args.prediction_task, x_train[0])
    print(f"Prediction task: {prediction_task}")

    output_cat, output_real = infer_output_spaces(
        x_train,
        x_valid,
        x_test,
        prediction_task,
    )
    print("Categorical outputs:", output_cat)
    print("Numerical outputs:", output_real)

    debug_smoke_test_config = {
        "hid": 128,
        "layers": 2,
        "lr": 1e-3,
        "aggregation": "sum",
        "batch_size": 128,
    }

    def train_evaluate(config: dict) -> dict:
        try:
            result, _ = train_object_hgnn(
                config,
                x_train,
                x_valid,
                output_cat,
                output_real,
                edge_types,
                device,
                NUM_EPOCHS,
            )
        except Exception as exc:
            print("\ntrain_evaluate failed for config:")
            print(config)
            print(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise

        valid_loss = result.get("valid_loss")
        if valid_loss is None:
            raise ValueError(
                "train_evaluate did not receive 'valid_loss' from train_object_hgnn."
            )
        if not np.isfinite(valid_loss):
            raise ValueError(
                f"train_evaluate produced a non-finite valid_loss={valid_loss} "
                f"for config={config}"
            )
        return {"valid_loss": valid_loss}

    print("\nRunning smoke test before Ax optimize...")
    smoke_test_result = train_evaluate(debug_smoke_test_config)
    print("Smoke test result:")
    print(smoke_test_result)

    if args.smoke_only:
        print("Smoke-only mode selected. Skipping Ax optimization and final runs.")
        return

    best_parameters, values, experiment, _ = optimize(
        parameters=[
            {
                "name": "hid",
                "type": "choice",
                "values": [128],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": False,
            },
            {
                "name": "layers",
                "type": "choice",
                "values": [2, 4],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": False,
            },
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-4, 1e-2],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "aggregation",
                "type": "choice",
                "values": ["sum", "mean", "max"],
                "value_type": "str",
            },
            {
                "name": "batch_size",
                "type": "choice",
                "values": [16, 64, 128, 256, 512],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": False,
            },
        ],
        evaluation_function=train_evaluate,
        objective_name="valid_loss",
        arms_per_trial=1,
        minimize=True,
        random_seed=123,
        total_trials=args.trials,
    )

    print(best_parameters)
    means, _ = values
    print(means)
    print(experiment)

    search_results = exp_to_df(experiment).sort_values(by="valid_loss")
    search_results.to_csv(
        os.path.join(results_dir, "hyp_params_search_object_hetero.csv"),
        index=False,
    )
    print(f"Saved object-hetero hyperparameter search results to: {results_dir}")

    test_multi(
        best_parameters,
        x_train,
        x_valid,
        x_test,
        output_cat,
        output_real,
        edge_types,
        device,
        NUM_EPOCHS,
        args.runs,
        results_dir,
    )


if __name__ == "__main__":
    main()
