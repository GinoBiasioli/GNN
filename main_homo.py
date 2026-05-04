import argparse
import torch
import numpy as np
import pandas as pd
import os
import json 
import platform
import time
from pathlib import Path
import random
import logging
import sys
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, ModuleDict
from torch_geometric.nn import GATv2Conv, global_mean_pool
import warnings
import traceback
from ax.service.managed_loop import optimize
import pandas
import torch.nn as nn
from copy import deepcopy
from datetime import datetime
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from ax.service.utils.report_utils import exp_to_df

DEFAULT_DATASET = "bpi_2012"
DEFAULT_TRIALS = 10
DEFAULT_RUNS = 10


# %%

def resolve_project_root():
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


root_path = resolve_project_root()


def resolve_dir(env_var_name, default_path):
    env_value = os.environ.get(env_var_name)
    if env_value:
        return os.path.abspath(os.path.expanduser(env_value))
    return default_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a homogeneous GNN on prefix graphs."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=os.environ.get("THESIS_DATASET", DEFAULT_DATASET),
        help="Dataset folder under data/datasets/hom_graphs.",
    )
    parser.add_argument(
        "--prediction-task",
        choices=["next_activity", "next_event"],
        default="next_activity",
        help=(
            "next_activity uses y_activity only. next_event uses y_activity, "
            "y_resource, and y_time."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        "--smoke-test",
        action="store_true",
        help="Run the smoke-test configuration before the selected workflow.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the smoke-test configuration and skip search/final repeated runs.",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Load the first row from the saved hyperparameter CSV and skip Ax search.",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Run Ax search, save the hyperparameter CSV, and skip final repeated runs.",
    )
    args, _ = parser.parse_known_args()
    return args


def format_elapsed_time(elapsed_seconds):
    return (
        f"{elapsed_seconds:.2f}s "
        f"({elapsed_seconds / 60:.2f} min, {elapsed_seconds / 3600:.2f} h)"
    )


def get_machine_report(device):
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": None,
        "gpu_total_memory_gb": None,
        "gpu_capability": None,
    }

    if torch.cuda.is_available():
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(gpu_index)
        report.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024**3), 3),
                "gpu_capability": f"{props.major}.{props.minor}",
            }
        )

    return report


def print_runtime_report(stage, elapsed_seconds, machine_report):
    print("\nRuntime report")
    print(f"Stage: {stage}")
    print(f"Elapsed: {format_elapsed_time(elapsed_seconds)}")
    print(f"Device: {machine_report['device']}")
    print(f"GPU: {machine_report['gpu_name']}")
    print(f"GPU memory (GB): {machine_report['gpu_total_memory_gb']}")
    print(f"Torch: {machine_report['torch_version']}")
    print(f"CUDA available: {machine_report['cuda_available']}")
    print(f"CUDA version: {machine_report['torch_cuda_version']}")
    print(f"CPU count: {machine_report['cpu_count']}")
    print(f"Platform: {machine_report['platform']}")


def append_runtime_report(
    results_dir,
    dataset,
    prediction_task,
    stage,
    elapsed_seconds,
    machine_report,
):
    record = {
        "dataset": dataset,
        "prediction_task": prediction_task,
        "stage": stage,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "elapsed_minutes": round(elapsed_seconds / 60, 4),
        "elapsed_hours": round(elapsed_seconds / 3600, 4),
        **machine_report,
    }
    path = os.path.join(results_dir, "runtime_report_homo.csv")
    pd.DataFrame([record]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )


args = parse_args()

pd.set_option("display.max_columns", None)

data_dir_processed = resolve_dir(
    "THESIS_PROCESSED_DIR",
    os.path.join(root_path, "data", "datasets", "processed"),
)
data_dir_graphs = resolve_dir(
    "THESIS_GRAPHS_DIR",
    os.path.join(root_path, "data", "datasets", "hom_graphs"),
)
results_root_dir = resolve_dir(
    "THESIS_RESULTS_DIR",
    os.path.join(root_path, "results"),
)

print(root_path, data_dir_processed, data_dir_graphs, sep="\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
machine_report = get_machine_report(device)
print_runtime_report("machine_setup", 0.0, machine_report)



#%%
PATIENCE = 10
NUM_EPOCHS = args.epochs
TOT_TRIALS = args.trials

#%%


 
with open(os.path.join(root_path, "data", "dataset_features.json"), "r") as file:
    datasets_info = json.load(file)   

list(datasets_info.keys())

def resolve_dataset_name(available_datasets, default_dataset):
    env_dataset = os.environ.get("THESIS_DATASET")
    dataset_name = args.dataset or env_dataset or default_dataset

    if dataset_name not in available_datasets:
        available = ", ".join(sorted(available_datasets))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose one of: {available}"
        )

    return dataset_name


dataset = resolve_dataset_name(datasets_info.keys(), DEFAULT_DATASET)
print(f"Selected dataset: {dataset}")
print(f"Prediction task: {args.prediction_task}")
print(f"Results directory: {os.path.join(results_root_dir, dataset, 'homo')}")

results_dir = os.path.join(results_root_dir, dataset, "homo")
os.makedirs(results_dir, exist_ok=True)


#%%



with open(os.path.join(root_path, "data", "dataset_features.json"), "r") as file:
    dataset_info = json.load(file)[dataset]

dataset_info

categorical_columns = dataset_info["categorical"]
real_value_columns = dataset_info["numerical"]

activity_target_col = "concept:name" if "concept:name" in categorical_columns else "Activity"
required_categorical_targets = {activity_target_col, "org:resource"}
required_numerical_targets = {"time:timestamp"}

missing_categorical_targets = required_categorical_targets - set(categorical_columns)
missing_numerical_targets = required_numerical_targets - set(real_value_columns)

if missing_categorical_targets or missing_numerical_targets:
    raise ValueError(
        "The selected dataset does not contain all required prediction targets. "
        f"Missing categorical: {sorted(missing_categorical_targets)}; "
        f"missing numerical: {sorted(missing_numerical_targets)}."
    )

#%%
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)

#%%
data_dir_graphs

#%%

def load_dataset(name):
    path = os.path.join(data_dir_graphs, dataset, name)
    size = os.path.getsize(path) / (1024**3)
    print(f'\nImporting "{name}" ({size:.2f} GB)')
    loaded_data = torch.load(path, weights_only=False)

    summary = {
        "dataset": dataset,
        "file_name": name,
        "graphs_count": len(loaded_data),
        "file_size_gb": round(size, 4),
    }

    print(
        f'Imported "{name}": '
        f'{summary["graphs_count"]} graphs'
    )

    return loaded_data, summary

dataset_import_summaries = []

for name in tqdm(["train_set_homo.pt","validation_set_homo.pt","test_set_homo.pt"]):
    if name == "train_set_homo.pt":
        X_TRAIN, summary = load_dataset(name)
    elif name == "validation_set_homo.pt":
        X_VALID, summary = load_dataset(name)
    else:
        X_TEST, summary = load_dataset(name)

    dataset_import_summaries.append(summary)

dataset_import_summary_path = os.path.join(results_dir, "dataset_import_summary.csv")
pd.DataFrame(dataset_import_summaries).to_csv(
    dataset_import_summary_path,
    index=False,
)
print(f"Saved dataset import summary to: {dataset_import_summary_path}")
        
#%%

class HomoGNN(Module):
    def __init__(self, input_dim, output_cat, output_real, parameters):
        super().__init__()

        hid = parameters["hid"] #size of hidden embeddings
        layers = parameters["layers"] #number of GNN layers
        self.output_cat = output_cat
        self.output_real = output_real

        self.convs = ModuleList()

        #first layer: input_dim -> hid
        self.convs.append(
            GATv2Conv(
                in_channels=input_dim,
                out_channels=hid,
                heads=1,
                concat=False,
                add_self_loops=False,
                residual=False
            )
        )

        # hidden layers: hid -> hid
        for _ in range(layers - 1):
            self.convs.append(
                GATv2Conv(
                    in_channels=hid,
                    out_channels=hid,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                    residual=False
                )
            )

        self.fc_cat = ModuleDict(
            {name: Linear(hid, output_cat[name]) for name in output_cat}
        )
        self.fc_real = ModuleDict(
            {name: Linear(hid, 1) for name in output_real}
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch.batch)
        out = {}

        for name, head in self.fc_cat.items():
            out[name] = head(x)
        for name, head in self.fc_real.items():
            out[name] = head(x)

        return out

def train_homo_gnn(config, output_cat, output_real):
    print(config)

    net = HomoGNN(
        input_dim=X_TRAIN[0].x.shape[1],
        output_cat=output_cat,
        output_real=output_real,
        parameters=config,
    )

    net = net.to(device)

    criterion_cat = nn.CrossEntropyLoss()
    criterion_real = nn.L1Loss()

    train_loader = DataLoader(
        X_TRAIN,
        batch_size=config["batch_size"],
        shuffle=True
    )
    valid_loader = DataLoader(
        X_VALID,
        batch_size=config["batch_size"],
        shuffle=False
    )

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config["lr"]
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(PATIENCE / 2)
    )

    best_model = None
    best_loss = None
    best_f1 = 0
    best_eval_details = None
    patience = PATIENCE
    pat_count = 0

    torch.cuda.empty_cache()

    for epoch in tqdm(range(0, NUM_EPOCHS)):


        # TRAIN
        net.train()

        for _, x in enumerate(train_loader):
            x = x.to(device)

            labels_cat = {
                "Activity": x.y_activity.view(-1).long(),
                "org:resource": x.y_resource.view(-1).long(),
            }
            labels_real = {
                "time:timestamp": x.y_time.view(-1, 1).float(),
            }

            optimizer.zero_grad()

            outputs = net(x)

            total_loss = torch.tensor(0.0, device=device)
            for name in output_cat:
                total_loss = total_loss + criterion_cat(outputs[name], labels_cat[name])
            for name in output_real:
                total_loss = total_loss + criterion_real(outputs[name], labels_real[name])

            total_loss.backward()
            optimizer.step()


        # VALIDATION
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
            for i, x in enumerate(valid_loader):
                x = x.to(device)

                labels_cat = {
                    "Activity": x.y_activity.view(-1).long(),
                    "org:resource": x.y_resource.view(-1).long(),
                }
                labels_real = {
                    "time:timestamp": x.y_time.view(-1, 1).float(),
                }

                outputs = net(x)

                val_batch_loss = torch.tensor(0.0, device=device)
                for name in output_cat:
                    current_loss = criterion_cat(outputs[name], labels_cat[name])
                    val_batch_loss = val_batch_loss + current_loss
                    running_cat_loss[name].append(current_loss.detach())
                for name in output_real:
                    current_loss = criterion_real(outputs[name], labels_real[name])
                    val_batch_loss = val_batch_loss + current_loss
                    running_real_loss[name].append(current_loss.detach())
                    avg_mae[name].append(
                        torch.mean(torch.abs(outputs[name] - labels_real[name])).detach()
                    )
                running_total_loss.append(val_batch_loss.detach())

                for name in output_cat:
                    preds = torch.argmax(torch.softmax(outputs[name], dim=1), dim=1)
                    predictions_categorical[name].append(preds)
                    target_categorical[name].append(labels_cat[name])
                for name in output_real:
                    prediction_numerical[name].append(outputs[name].detach())
                    target_numerical[name].append(labels_real[name].detach())

        val_loss = torch.stack(running_total_loss).mean()

        lr_scheduler.step(val_loss)

        for name in output_cat:
            predictions_categorical[name] = torch.cat(predictions_categorical[name])
            target_categorical[name] = torch.cat(target_categorical[name])
        for name in output_real:
            prediction_numerical[name] = torch.cat(prediction_numerical[name]).view(-1)
            target_numerical[name] = torch.cat(target_numerical[name]).view(-1)
            avg_mae[name] = torch.stack(avg_mae[name]).mean()

        avg_cat_loss = {
            name: torch.stack(running_cat_loss[name]).mean()
            for name in output_cat
        }
        avg_real_loss = {
            name: torch.stack(running_real_loss[name]).mean()
            for name in output_real
        }

        macro_f1s = {
            name: multiclass_f1_score(
                predictions_categorical[name],
                target_categorical[name],
                num_classes=output_cat[name],
                average="macro"
            )
            for name in output_cat
        }

        accuracy = {
            name: multiclass_accuracy(
                predictions_categorical[name],
                target_categorical[name],
                num_classes=output_cat[name],
            )
            for name in output_cat
        }

        f1_activity = macro_f1s["Activity"]
        eval_details = {
            "valid_loss": val_loss.item(),
            **{f"valid_{name}_loss": avg_cat_loss[name].item() for name in avg_cat_loss},
            **{f"valid_{name}_loss": avg_real_loss[name].item() for name in avg_real_loss},
            **{f"valid_{name}_acc": accuracy[name].item() for name in accuracy},
            **{f"valid_{name}_macroF1": macro_f1s[name].item() for name in macro_f1s},
            **{f"valid_{name}_MAE": avg_mae[name].item() for name in avg_mae},
        }

        if epoch == 0:
            best_model = deepcopy(net)
            best_loss = val_loss
            best_f1 = f1_activity
            best_eval_details = eval_details.copy()
            pat_count = 0

            print("/" * 10)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print("Acc", accuracy)
            print("F1s", macro_f1s)
            print("Cat loss", avg_cat_loss)
            print("Real loss", avg_real_loss)
            print("MAE", avg_mae)
            print(
                f"Patience {pat_count}/{patience}, "
                f"val loss {val_loss.item()} "
                f"current_lr {lr_scheduler.get_last_lr()}, "
                f"curr_best_activity_F1 {best_f1}"
            )
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(net)
                best_eval_details = eval_details.copy()
                pat_count = 0
            else:
                pat_count += 1

            if best_f1 < f1_activity:
                best_f1 = f1_activity

            print("/" * 10)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print("Acc", accuracy)
            print("F1s", macro_f1s)
            print("Cat loss", avg_cat_loss)
            print("Real loss", avg_real_loss)
            print("MAE", avg_mae)
            print(
                f"Patience {pat_count}/{patience}, "
                f"val loss {val_loss.item()} "
                f"current_lr {lr_scheduler.get_last_lr()}, "
                f"curr_best_activity_F1 {best_f1}"
            )

            if pat_count == patience:
                return (best_eval_details, best_model)

    print("Acc", accuracy)
    print("F1s", macro_f1s)
    print("Cat loss", avg_cat_loss)
    print("Real loss", avg_real_loss)
    print("MAE", avg_mae)
    print(
        f"Patience {pat_count}/{patience}, "
        f"val loss {val_loss.item()} "
        f"current_lr {lr_scheduler.get_last_lr()}, "
        f"curr_best_activity_F1 {best_f1}"
    )

    return (best_eval_details, best_model)

def test_homo_gnn(net, output_cat, output_real):
    criterion_cat = nn.CrossEntropyLoss()
    criterion_real = nn.L1Loss()

    predictions_categorical = {name: [] for name in output_cat}
    target_categorical = {name: [] for name in output_cat}
    prediction_numerical = {name: [] for name in output_real}
    target_numerical = {name: [] for name in output_real}
    avg_mae = {name: [] for name in output_real}
    running_cat_loss = {name: [] for name in output_cat}
    running_real_loss = {name: [] for name in output_real}

    running_total_loss = []

    test_loader = DataLoader(X_TEST, batch_size=128, shuffle=False)

    net.eval()
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.to(device)

            labels_cat = {
                "Activity": x.y_activity.view(-1).long(),
                "org:resource": x.y_resource.view(-1).long(),
            }
            labels_real = {
                "time:timestamp": x.y_time.view(-1, 1).float(),
            }
            outputs = net(x)

            loss = torch.tensor(0.0, device=device)
            for name in output_cat:
                current_loss = criterion_cat(outputs[name], labels_cat[name])
                loss = loss + current_loss
                running_cat_loss[name].append(current_loss.detach())
            for name in output_real:
                current_loss = criterion_real(outputs[name], labels_real[name])
                loss = loss + current_loss
                running_real_loss[name].append(current_loss.detach())
                avg_mae[name].append(
                    torch.mean(torch.abs(outputs[name] - labels_real[name])).detach()
                )
            running_total_loss.append(loss.detach())

            for name in output_cat:
                preds = torch.argmax(torch.softmax(outputs[name], dim=1), dim=1)
                predictions_categorical[name].append(preds)
                target_categorical[name].append(labels_cat[name])
            for name in output_real:
                prediction_numerical[name].append(outputs[name].detach())
                target_numerical[name].append(labels_real[name].detach())

    for name in output_cat:
        predictions_categorical[name] = torch.cat(predictions_categorical[name])
        target_categorical[name] = torch.cat(target_categorical[name])
    for name in output_real:
        prediction_numerical[name] = torch.cat(prediction_numerical[name]).view(-1)
        target_numerical[name] = torch.cat(target_numerical[name]).view(-1)
        avg_mae[name] = torch.stack(avg_mae[name]).mean()

    avg_cat_loss = {
        name: torch.stack(running_cat_loss[name]).mean()
        for name in output_cat
    }
    avg_real_loss = {
        name: torch.stack(running_real_loss[name]).mean()
        for name in output_real
    }

    macro_f1s = {
        name: multiclass_f1_score(
            predictions_categorical[name],
            target_categorical[name],
            num_classes=output_cat[name],
            average="macro"
        )
        for name in output_cat
    }

    accuracy = {
        name: multiclass_accuracy(
            predictions_categorical[name],
            target_categorical[name],
            num_classes=output_cat[name],
        )
        for name in output_cat
    }

    Average_total_loss = torch.stack(running_total_loss).mean()

    res = (
        {f"{k}_loss": avg_cat_loss[k].item() for k in avg_cat_loss}
        | {f"{k}_loss": avg_real_loss[k].item() for k in avg_real_loss}
        | {f"{k}_acc": accuracy[k].item() for k in accuracy}
        | {f"{k}_macroF1": macro_f1s[k].item() for k in macro_f1s}
        | {f"{k}_MAE": avg_mae[k].item() for k in avg_mae}
        | {"AVG_total_loss": Average_total_loss.item()}
    )

    print(res)

    return res


all_activity_labels = [int(g.y_activity.item()) for g in X_TRAIN + X_VALID + X_TEST]

outputcat = {
    "Activity": max(all_activity_labels) + 1,
}
outputreal = {}

if args.prediction_task == "next_event":
    all_resource_labels = [int(g.y_resource.item()) for g in X_TRAIN + X_VALID + X_TEST]
    outputcat["org:resource"] = max(all_resource_labels) + 1
    outputreal["time:timestamp"] = 1

print(outputcat)
print(outputreal)



logging.getLogger("root").setLevel(logging.ERROR)


warnings.filterwarnings("ignore", category=UserWarning)


DEBUG_SMOKE_TEST_CONFIG = {
    "hid": 128,
    "layers": 2,
    "lr": 1e-3,
    "batch_size": 128,
}


def train_evaluate(config):
    try:
        res, _ = train_homo_gnn(
            config,
            output_cat=outputcat,
            output_real=outputreal
        )
    except Exception as exc:
        print("\ntrain_evaluate failed for config:")
        print(config)
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise

    valid_loss = res.get("valid_loss")
    if valid_loss is None:
        raise ValueError(
            "train_evaluate did not receive 'valid_loss' from train_homo_gnn."
        )

    if not np.isfinite(valid_loss):
        raise ValueError(
            f"train_evaluate produced a non-finite valid_loss={valid_loss} "
            f"for config={config}"
        )

    return {"valid_loss": valid_loss}


def load_best_hyperparameters(search_results_path):
    if not os.path.exists(search_results_path):
        raise FileNotFoundError(
            f"Could not find saved hyperparameter search results: {search_results_path}. "
            "Run without --skip-search first."
        )

    search_results = pd.read_csv(search_results_path)
    if search_results.empty:
        raise ValueError(f"Hyperparameter search file is empty: {search_results_path}")

    if "valid_loss" in search_results.columns:
        search_results = search_results.sort_values(by="valid_loss")

    best_row = search_results.iloc[0]
    required_columns = ["hid", "layers", "lr", "batch_size"]
    missing_columns = [col for col in required_columns if col not in best_row.index]
    if missing_columns:
        raise ValueError(
            f"Hyperparameter search file is missing columns {missing_columns}: "
            f"{search_results_path}"
        )

    return {
        "hid": int(best_row["hid"]),
        "layers": int(best_row["layers"]),
        "lr": float(best_row["lr"]),
        "batch_size": int(best_row["batch_size"]),
    }


search_results_path = os.path.join(results_dir, "hyp_params_search_homo.csv")

if args.smoke_test or args.smoke_only:
    print("\nRunning smoke test...")
    smoke_test_result = train_evaluate(DEBUG_SMOKE_TEST_CONFIG)
    print("Smoke test result:")
    print(smoke_test_result)

if args.smoke_only:
    print("Smoke-only mode selected. Skipping Ax optimization and final runs.")
    sys.exit(0)

if args.skip_search:
    print(f"Skipping Ax search. Loading best hyperparameters from: {search_results_path}")
else:
    search_start_time = time.perf_counter()
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "hid",
                "type": "choice",
                "values": [128],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": False
            },
            {
                "name": "layers",
                "type": "choice",
                "values": [2, 4],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": False
            },
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-4, 1e-2],
                "value_type": "float",
                "log_scale": True
            },
            {
                "name": "batch_size",
                "type": "choice",
                "values": [128, 256, 512],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": False
            },
        ],
        evaluation_function=train_evaluate,
        objective_name="valid_loss",
        arms_per_trial=1,
        minimize=True,
        random_seed=123,
        total_trials=TOT_TRIALS
    )
    search_elapsed_seconds = time.perf_counter() - search_start_time

    print(best_parameters)
    means, covariances = values
    print(means)
    print(experiment)

    results = exp_to_df(experiment)
    results = results.sort_values(by="valid_loss")
    results.to_csv(search_results_path, sep=",", index=False)
    print(f"Saved homogeneous hyperparameter search results to: {search_results_path}")
    print_runtime_report(
        "hyperparameter_search",
        search_elapsed_seconds,
        machine_report,
    )
    append_runtime_report(
        results_dir,
        dataset,
        args.prediction_task,
        "hyperparameter_search",
        search_elapsed_seconds,
        machine_report,
    )

best_parameters = load_best_hyperparameters(search_results_path)
print("Best hyperparameters loaded from CSV first row:")
print(best_parameters)

if args.search_only:
    print("Search-only mode selected. Skipping final repeated runs.")
    sys.exit(0)
def create_df(results):
    res = {}

    for k in results[0]:
        res[k] = [x[k] for x in results]

    res = pandas.DataFrame(data=res)

    return res, res.mean(), res.std()


def test_multi(config, outputcat, outputreal, num_runs=10):
    res = []


    save_path = results_dir
    os.makedirs(save_path, exist_ok=True)
    for i in range(num_runs):
        print(f"Run {i}")

        _, net = train_homo_gnn(
            config,
            outputcat,
            outputreal,
        )

        res.append(
            test_homo_gnn(
                net,
                outputcat,
                outputreal,
            )
        )

        print("RES:")
        print(res[-1])

    results_table, means, stds = create_df(res)
    results_table.to_csv(os.path.join(save_path, "results_homo.csv"), sep=",", index=False)

    pd.DataFrame(data={"mean": means, "std": stds}).to_csv(
    os.path.join(save_path, "mean_and_stds_homo.csv"),
    sep=","
    )

    print(pd.DataFrame(data={"mean": means, "std": stds}))
    print(f"Saved final homogeneous evaluation results to: {save_path}")

    return res


training_start_time = time.perf_counter()
test_multi(best_parameters, outputcat, outputreal, args.runs)
training_elapsed_seconds = time.perf_counter() - training_start_time
print_runtime_report(
    "final_training_runs",
    training_elapsed_seconds,
    machine_report,
)
append_runtime_report(
    results_dir,
    dataset,
    args.prediction_task,
    "final_training_runs",
    training_elapsed_seconds,
    machine_report,
)
