import os
import json
from math import log
from os.path import dirname
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


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


data_dir_processed = resolve_dir(
    "THESIS_PROCESSED_DIR",
    os.path.join(root_path, "data", "datasets", "processed"),
)
data_dir_graphs = resolve_dir(
    "THESIS_GRAPHS_DIR",
    os.path.join(root_path, "data", "datasets", "hom_graphs"),
)



pd.set_option("display.max_columns", None)

print(root_path, data_dir_processed, data_dir_graphs, sep="\n")



#dataset = "BPI20_RequestForPayment"
dataset = "bpi_2012"
#dataset = "bpi_2013"
#dataset = "sp2020"




EXTRA_CATEGORICAL_COLUMNS = []
EXTRA_NUMERICAL_COLUMNS = []

UNDIRECT_TRANSFORMATION = ToUndirected()



#with open(r"G:\\Mi unidad\\Thesis\\SephigraphV2-main\\SephigraphV2-main\\data\\dataset_features.json") as file:
with open(os.path.join(root_path, "data", "dataset_features.json"), "r") as file: 
    datasets_info = json.load(file)

dataset_info = datasets_info[dataset]

categorical_columns = list(dataset_info["categorical"]) + EXTRA_CATEGORICAL_COLUMNS
real_value_columns = list(dataset_info["numerical"]) + EXTRA_NUMERICAL_COLUMNS

print("\nCategorical columns:")
print(categorical_columns)
print("\nNumerical columns:")
print(real_value_columns)



#tab_all = pd.read_csv(f"{data_dir_processed}{dataset}/{dataset}_processed_all.csv")
#tab_train = pd.read_csv(f"{data_dir_processed}{dataset}/{dataset}_processed_train.csv")
#tab_valid = pd.read_csv(f"{data_dir_processed}{dataset}/{dataset}_processed_valid.csv")
#tab_test = pd.read_csv(f"{data_dir_processed}{dataset}/{dataset}_processed_test.csv")

tab_all = pd.read_csv(os.path.join(data_dir_processed, dataset, f"{dataset}_processed_all.csv"))
tab_train = pd.read_csv(os.path.join(data_dir_processed, dataset, f"{dataset}_processed_train.csv"))
tab_valid = pd.read_csv(os.path.join(data_dir_processed, dataset, f"{dataset}_processed_valid.csv"))
tab_test = pd.read_csv(os.path.join(data_dir_processed, dataset, f"{dataset}_processed_test.csv"))

categorical_columns = [c for c in categorical_columns if c in tab_all.columns]
real_value_columns = [c for c in real_value_columns if c in tab_all.columns]

print("\nAvailable categorical columns:")
print(categorical_columns)
print("\nAvailable numerical columns:")
print(real_value_columns)



for k in categorical_columns:
    tab_all[k] = tab_all[k].astype("object")
    tab_train[k] = tab_train[k].astype("object")
    tab_valid[k] = tab_valid[k].astype("object")
    tab_test[k] = tab_test[k].astype("object")

for k in real_value_columns:
    tab_all[k] = pd.to_numeric(tab_all[k], errors="coerce")
    tab_train[k] = pd.to_numeric(tab_train[k], errors="coerce")
    tab_valid[k] = pd.to_numeric(tab_valid[k], errors="coerce")
    tab_test[k] = pd.to_numeric(tab_test[k], errors="coerce")

tab_train["CaseID"] = tab_train["CaseID"].astype(np.str_)
tab_valid["CaseID"] = tab_valid["CaseID"].astype(np.str_)
tab_test["CaseID"] = tab_test["CaseID"].astype(np.str_)


def fit_one_hot_encoders(dataset_df: pd.DataFrame, cat_cols):
    encoders = {}
    for col in cat_cols:
        values = normalize_categorical_series(dataset_df[col]).values.reshape(-1, 1)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(values)
        encoders[col] = encoder
    return encoders


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("object").where(series.notna(), "__MISSING__")
    return normalized.astype(np.str_)


def normalize_categorical_value(value) -> str:
    if pd.isna(value):
        return "__MISSING__"
    return str(value)

ONE_HOT_ENCODERS = fit_one_hot_encoders(tab_all, categorical_columns)

ACTIVITY_LABEL_COL = "concept:name" if "concept:name" in tab_all.columns else "Activity"
activity_label_values = normalize_categorical_series(tab_all[ACTIVITY_LABEL_COL]).values.reshape(-1, 1)
ACTIVITY_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ACTIVITY_ENCODER.fit(activity_label_values)
ACTIVITY_CLASSES = list(ACTIVITY_ENCODER.categories_[0])
ACTIVITY_TO_INDEX = {cls_name: idx for idx, cls_name in enumerate(ACTIVITY_CLASSES)}

resource_label_values = normalize_categorical_series(tab_all["org:resource"]).values.reshape(-1, 1)
RESOURCE_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
RESOURCE_ENCODER.fit(resource_label_values)
RESOURCE_CLASSES = list(RESOURCE_ENCODER.categories_[0])
RESOURCE_TO_INDEX = {cls_name: idx for idx, cls_name in enumerate(RESOURCE_CLASSES)}


def get_case_ids(df: pd.DataFrame):
    return list(df["CaseID"].unique())

def add_new_timestamp(trace: pd.DataFrame) -> pd.DataFrame:
    trace2 = trace.copy()
    if "time:timestamp" not in trace2.columns:
        return trace2
    times = list(trace2["time:timestamp"].copy())
    if len(times) == 0:
        return trace2
    first_time = times[0]
    for i in range(1, len(times)):
        times[i] = times[i] - first_time
    times[0] = 0.0
    trace2["time:timestamp"] = times
    return trace2



def encode_trace_events(trace: pd.DataFrame, cat_cols, num_cols, encoders) -> np.ndarray:
    #transforms each event in a case into a numerical feature vector
    blocks = []
    for col in cat_cols:
        vals = normalize_categorical_series(trace[col]).values.reshape(-1, 1)
        encoded = encoders[col].transform(vals).astype(np.float32)
        blocks.append(encoded)
    for col in num_cols:
        vals = pd.to_numeric(trace[col], errors="coerce").fillna(0.0).values.astype(np.float32).reshape(-1, 1)
        blocks.append(vals)
    x = np.concatenate(blocks, axis=1).astype(np.float32)
    return x



def build_event_chain_edges(prefix_len: int) -> torch.Tensor:
    if prefix_len <= 1: #no edges possible0
        return torch.empty((2, 0), dtype=torch.long)
    src = list(range(prefix_len - 1))
    dst = list(range(1, prefix_len))
    return torch.tensor([src, dst], dtype=torch.long)


def get_next_activity_label(trace: pd.DataFrame, next_event_idx: int) -> torch.Tensor:
    next_activity = normalize_categorical_value(trace.iloc[next_event_idx][ACTIVITY_LABEL_COL])
    class_idx = ACTIVITY_TO_INDEX[next_activity]
    return torch.tensor([class_idx], dtype=torch.long)


def get_next_resource_label(trace: pd.DataFrame, next_event_idx: int) -> torch.Tensor:
    next_resource = normalize_categorical_value(trace.iloc[next_event_idx]["org:resource"])
    class_idx = RESOURCE_TO_INDEX[next_resource]
    return torch.tensor([class_idx], dtype=torch.long)


def get_next_timestamp_label(trace: pd.DataFrame, next_event_idx: int) -> torch.Tensor:
    next_timestamp = pd.to_numeric(
        pd.Series([trace.iloc[next_event_idx]["time:timestamp"]]),
        errors="coerce",
    ).fillna(0.0).iloc[0]
    return torch.tensor([float(next_timestamp)], dtype=torch.float32)


def build_prefixes_graph_from_trace_homogeneous(trace, cat_features, real_features):
    X_homo = []
    trace = add_new_timestamp(trace)
    full_x = encode_trace_events(trace, cat_features, real_features, ONE_HOT_ENCODERS)

    for prefix in range(1, len(trace) - 1):
        prefix_len = prefix + 1
        x_prefix = torch.tensor(full_x[:prefix_len], dtype=torch.float32)
        edge_index = build_event_chain_edges(prefix_len)
        y_activity = get_next_activity_label(trace, prefix + 1)
        y_resource = get_next_resource_label(trace, prefix + 1)
        y_time = get_next_timestamp_label(trace, prefix + 1)

        G = Data(x=x_prefix, edge_index=edge_index)
        G = UNDIRECT_TRANSFORMATION(G)
        G.y = y_activity
        G.y_activity = y_activity
        G.y_resource = y_resource
        G.y_time = y_time
        G.prefix_len = prefix_len
        G.case_len = len(trace)
        G.num_node_features_ = x_prefix.shape[1]

        X_homo.append(G)

    return X_homo


def build_split_graphs_homo(df_split, split_name, cat_features, real_features):
    print(f"\nPreparing {split_name} dataset...")
    case_ids = get_case_ids(df_split)
    X_split_homo = []

    for case_id in tqdm(case_ids):
        trace = (
            df_split.query(f"CaseID == '{case_id}'")
            .reset_index(drop=True)
            .drop(columns=["CaseID"])
        )

        if len(trace) > 2:
            graphs = build_prefixes_graph_from_trace_homogeneous(trace, cat_features, real_features)
            X_split_homo.extend(graphs)

    print(f"{split_name} graphs created: {len(X_split_homo)}")
    return X_split_homo


X_train_homo = build_split_graphs_homo(tab_train, "training", categorical_columns, real_value_columns)
X_val_homo = build_split_graphs_homo(tab_valid, "validation", categorical_columns, real_value_columns)
X_test_homo = build_split_graphs_homo(tab_test, "test", categorical_columns, real_value_columns)



#save_dir = f"{data_dir_graphs}{dataset}"
#os.makedirs(save_dir, exist_ok=True)

#torch.save(X_train_homo, f"{save_dir}/train_set_homo.pt")
#torch.save(X_val_homo, f"{save_dir}/validation_set_homo.pt")
#torch.save(X_test_homo, f"{save_dir}/test_set_homo.pt")


save_dir = os.path.join(data_dir_graphs, dataset)
os.makedirs(save_dir, exist_ok=True)

torch.save(X_train_homo, os.path.join(save_dir, "train_set_homo.pt"))
torch.save(X_val_homo, os.path.join(save_dir, "validation_set_homo.pt"))
torch.save(X_test_homo, os.path.join(save_dir, "test_set_homo.pt"))

#print("\nSaved files:")
#print(f"{save_dir}/train_set_homo.pt")
#print(f"{save_dir}/validation_set_homo.pt")
#print(f"{save_dir}/test_set_homo.pt")

print("\nSaved files:")
print(os.path.join(save_dir, "train_set_homo.pt"))
print(os.path.join(save_dir, "validation_set_homo.pt"))
print(os.path.join(save_dir, "test_set_homo.pt"))


if len(X_train_homo) > 0:
    g = X_train_homo[1053]
    print("\nExample graph:")
    print(g)
    print("x shape:", g.x.shape)
    print("edge_index shape:", g.edge_index.shape)
    print("y:", g.y)
    print("prefix_len:", g.prefix_len)
