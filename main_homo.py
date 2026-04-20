import torch
import numpy as np
import pandas as pd
import os
import json 
from pathlib import Path
import random
import logging
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
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from ax.service.utils.report_utils import exp_to_df



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



#%%
PATIENCE = 10
NUM_EPOCHS = 50
TOT_TRIALS = 10

#%%


 
with open(os.path.join(root_path, "data", "dataset_features.json"), "r") as file:
    datasets_info = json.load(file)   

list(datasets_info.keys())

#dataset = "BPI20_RequestForPayment"
#dataset = "bpi_2012"
#dataset = "bpi_2013"
#dataset = "sp2020"
dataset = "tiny_sp2020"


#%%



with open(os.path.join(root_path, "data", "dataset_features.json"), "r") as file:
    dataset_info = json.load(file)[dataset]

dataset_info

categorical_columns = dataset_info["categorical"]
real_value_columns = dataset_info["numerical"]

required_categorical_targets = {"Activity", "org:resource"}
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
    print(f"\nLoading {name} ({size:.2f} GB)")
    return torch.load(path, weights_only=False)

for name in tqdm(["train_set_homo.pt","validation_set_homo.pt","test_set_homo.pt"]):
    if name == "train_set_homo.pt":
        X_TRAIN = load_dataset(name)
    elif name == "validation_set_homo.pt":
        X_VALID = load_dataset(name)
    else:
        X_TEST = load_dataset(name)
        
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
all_resource_labels = [int(g.y_resource.item()) for g in X_TRAIN + X_VALID + X_TEST]

outputcat = {
    "Activity": max(all_activity_labels) + 1,
    "org:resource": max(all_resource_labels) + 1,
}
outputreal = {"time:timestamp": 1}

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


print("\nRunning smoke test before Ax optimize...")
smoke_test_result = train_evaluate(DEBUG_SMOKE_TEST_CONFIG)
print("Smoke test result:")
print(smoke_test_result)


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

print(best_parameters)
means, covariances = values
print(means)
print(experiment)

### 


results = exp_to_df(experiment)
results = results.sort_values(by="valid_loss")

#if not os.path.isdir(f"results/{dataset}"):
#    os.mkdir(f"results/{dataset}")

#results.to_csv(f"results/{dataset}/hyp_params_search.csv", sep=",", index=False)

results_dir = os.path.join(results_root_dir, dataset)
os.makedirs(results_dir, exist_ok=True)
results.to_csv(os.path.join(results_dir, "hyp_params_search.csv"), sep=",", index=False)
def create_df(results):
    res = {}

    for k in results[0]:
        res[k] = [x[k] for x in results]

    res = pandas.DataFrame(data=res)

    return res, res.mean(), res.std()


def test_multi(config, outputcat, outputreal, num_runs=10):
    res = []


    save_path = os.path.join(results_root_dir, dataset)
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
    results_table.to_csv(os.path.join(save_path, "results.csv"), sep=",", index=False)

    pd.DataFrame(data={"mean": means, "std": stds}).to_csv(
    os.path.join(save_path, "mean_and_stds.csv"),
    sep=","
    )

    print(pd.DataFrame(data={"mean": means, "std": stds}))

    return res


test_multi(best_parameters, outputcat, outputreal, 10)
