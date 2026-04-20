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
from torch.nn import Module, ModuleList, Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool
import warnings
from ax.service.managed_loop import optimize
import pandas
import torch.nn as nn
from copy import deepcopy
from torch_geometric.loader import DataLoader
from torch.nn.functional import l1_loss
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from ax.service.utils.report_utils import exp_to_df


import matplotlib.pyplot as plt
from collections import Counter


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




# %%
with open(os.path.join(root_path, "data", "dataset_features.json"), "r") as file:
    datasets_info = json.load(file)

print("Available datasets:", list(datasets_info.keys()))

#dataset = "BPI20_RequestForPayment"
dataset = "bpi_2012"
#dataset = "bpi_2013"
#dataset = "sp2020"


# %%
with open(os.path.join(root_path, "data", "dataset_features.json"), "r") as file:
    dataset_info = json.load(file)[dataset]

print("\nDataset info:")
print(dataset_info)

categorical_columns = dataset_info["categorical"]
real_value_columns = dataset_info["numerical"]

print("\nCategorical columns:", categorical_columns)
print("Numerical columns:", real_value_columns)


# %%
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)


# %%
print("\nGraph directory:")
print(data_dir_graphs)


# %%
def load_dataset(name):
    path = os.path.join(data_dir_graphs, dataset, name)
    size = os.path.getsize(path) / (1024**3)
    print(f"\nLoading {name} ({size:.2f} GB)")
    return torch.load(path, weights_only=False)

for name in tqdm(["train_set_homo.pt", "validation_set_homo.pt", "test_set_homo.pt"]):
    if name == "train_set_homo.pt":
        X_TRAIN = load_dataset(name)
    elif name == "validation_set_homo.pt":
        X_VALID = load_dataset(name)
    else:
        X_TEST = load_dataset(name)


# %%
# =============================================================================
# BASIC CHECKS
# =============================================================================

print("\n" + "=" * 80)
print("BASIC DATASET CHECKS")
print("=" * 80)

print(f"Train graphs: {len(X_TRAIN)}")
print(f"Valid graphs: {len(X_VALID)}")
print(f"Test graphs : {len(X_TEST)}")

sample_graph = X_TRAIN[0]
print("\nSample graph:")
print(sample_graph)

print("\nSample graph attributes:")
print(f"x shape        : {sample_graph.x.shape}")
print(f"edge_index shape: {sample_graph.edge_index.shape}")
print(f"y              : {sample_graph.y}")
print(f"y shape        : {sample_graph.y.shape if hasattr(sample_graph.y, 'shape') else 'scalar'}")


# %%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_graph_stats(graph):
    """
    Extract simple stats from one PyG homogeneous graph.
    """
    num_nodes = graph.num_nodes

    if hasattr(graph, "edge_index") and graph.edge_index is not None:
        num_edges = graph.edge_index.shape[1]
    else:
        num_edges = 0

    if hasattr(graph, "x") and graph.x is not None:
        num_features = graph.x.shape[1]
    else:
        num_features = 0

    if hasattr(graph, "y") and graph.y is not None:
        y_value = graph.y
        if isinstance(y_value, torch.Tensor):
            if y_value.numel() == 1:
                y_value = int(y_value.item())
            else:
                # just in case there is a weird shape
                y_value = y_value.view(-1).tolist()
        else:
            y_value = int(y_value)
    else:
        y_value = None

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": num_features,
        "y": y_value
    }


def summarize_split(graph_list, split_name):
    """
    Build a dataframe with graph-level stats for one split.
    """
    rows = []
    for g in tqdm(graph_list, desc=f"Processing {split_name}"):
        rows.append(get_graph_stats(g))

    df = pd.DataFrame(rows)

    print("\n" + "-" * 80)
    print(f"{split_name.upper()} SUMMARY")
    print("-" * 80)
    print(df.describe(include="all"))

    if "y" in df.columns:
        unique_y = sorted(df["y"].dropna().unique().tolist())
        print(f"\nUnique y classes in {split_name}: {len(unique_y)}")
        print(f"Classes: {unique_y[:50]}{' ...' if len(unique_y) > 50 else ''}")

    return df


def print_class_distribution(df, split_name, top_n=20):
    """
    Print class counts for y.
    """
    print("\n" + "-" * 80)
    print(f"{split_name.upper()} Y DISTRIBUTION")
    print("-" * 80)

    y_counts = df["y"].value_counts().sort_index()
    y_props = df["y"].value_counts(normalize=True).sort_index()

    dist_df = pd.DataFrame({
        "count": y_counts,
        "proportion": y_props
    })

    print(dist_df.head(top_n))
    if len(dist_df) > top_n:
        print(f"... showing first {top_n} classes out of {len(dist_df)}")

    return dist_df


def dataset_level_summary(train_df, valid_df, test_df):
    """
    Print a compact overall summary.
    """
    all_df = pd.concat(
        [
            train_df.assign(split="train"),
            valid_df.assign(split="valid"),
            test_df.assign(split="test")
        ],
        ignore_index=True
    )

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    print(f"Total graphs: {len(all_df)}")
    print(f"Total unique y classes: {all_df['y'].nunique()}")
    print(f"Feature dimension: {all_df['num_features'].iloc[0]}")

    print("\nNodes per graph:")
    print(all_df["num_nodes"].describe())

    print("\nEdges per graph:")
    print(all_df["num_edges"].describe())

    print("\nOverall y distribution:")
    overall_dist = all_df["y"].value_counts().sort_index()
    print(overall_dist)

    return all_df


def plot_histogram(series, title, xlabel, bins=30, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_bar(series, title, xlabel, ylabel="Count", save_path=None):
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_top_classes(df, split_name, top_n=20, save_path=None):
    counts = df["y"].value_counts().head(top_n)
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title(f"Top {top_n} most frequent y classes - {split_name}")
    plt.xlabel("y class")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# %%
# =============================================================================
# BUILD EDA TABLES
# =============================================================================

train_df = summarize_split(X_TRAIN, "train")
valid_df = summarize_split(X_VALID, "valid")
test_df  = summarize_split(X_TEST, "test")

train_y_dist = print_class_distribution(train_df, "train", top_n=50)
valid_y_dist = print_class_distribution(valid_df, "valid", top_n=50)
test_y_dist  = print_class_distribution(test_df, "test", top_n=50)

all_df = dataset_level_summary(train_df, valid_df, test_df)


# %%
# =============================================================================
# OPTIONAL: CHECK IF TRAIN/VALID/TEST HAVE SAME CLASS SPACE
# =============================================================================

train_classes = set(train_df["y"].dropna().unique())
valid_classes = set(valid_df["y"].dropna().unique())
test_classes = set(test_df["y"].dropna().unique())

print("\n" + "=" * 80)
print("CLASS SPACE CHECK")
print("=" * 80)

print("Classes in train:", len(train_classes))
print("Classes in valid:", len(valid_classes))
print("Classes in test :", len(test_classes))

print("\nClasses in valid but not in train:", sorted(valid_classes - train_classes))
print("Classes in test but not in train :", sorted(test_classes - train_classes))


# %%
# =============================================================================
# SIMPLE PLOTS
# =============================================================================

plots_dir = os.path.join(results_root_dir, dataset, "eda_plots")
os.makedirs(plots_dir, exist_ok=True)

# Nodes per graph
plot_histogram(
    train_df["num_nodes"],
    title=f"Nodes per graph - train - {dataset}",
    xlabel="Number of nodes",
    bins=30,
    save_path=os.path.join(plots_dir, "train_nodes_hist.png")
)

plot_histogram(
    valid_df["num_nodes"],
    title=f"Nodes per graph - valid - {dataset}",
    xlabel="Number of nodes",
    bins=30,
    save_path=os.path.join(plots_dir, "valid_nodes_hist.png")
)

plot_histogram(
    test_df["num_nodes"],
    title=f"Nodes per graph - test - {dataset}",
    xlabel="Number of nodes",
    bins=30,
    save_path=os.path.join(plots_dir, "test_nodes_hist.png")
)

# Edges per graph
plot_histogram(
    train_df["num_edges"],
    title=f"Edges per graph - train - {dataset}",
    xlabel="Number of edges",
    bins=30,
    save_path=os.path.join(plots_dir, "train_edges_hist.png")
)

# Y distribution
plot_top_classes(
    train_df,
    split_name="train",
    top_n=20,
    save_path=os.path.join(plots_dir, "train_top20_y.png")
)

plot_top_classes(
    valid_df,
    split_name="valid",
    top_n=20,
    save_path=os.path.join(plots_dir, "valid_top20_y.png")
)

plot_top_classes(
    test_df,
    split_name="test",
    top_n=20,
    save_path=os.path.join(plots_dir, "test_top20_y.png")
)


# %%
# =============================================================================
# SAVE SUMMARY TABLES
# =============================================================================

summary_dir = os.path.join(results_root_dir, dataset, "eda_tables")
os.makedirs(summary_dir, exist_ok=True)

train_df.to_csv(os.path.join(summary_dir, "train_graph_stats.csv"), index=False)
valid_df.to_csv(os.path.join(summary_dir, "valid_graph_stats.csv"), index=False)
test_df.to_csv(os.path.join(summary_dir, "test_graph_stats.csv"), index=False)

train_y_dist.to_csv(os.path.join(summary_dir, "train_y_distribution.csv"))
valid_y_dist.to_csv(os.path.join(summary_dir, "valid_y_distribution.csv"))
test_y_dist.to_csv(os.path.join(summary_dir, "test_y_distribution.csv"))

all_df.to_csv(os.path.join(summary_dir, "all_graph_stats.csv"), index=False)

print("\nSaved EDA tables to:")
print(summary_dir)

print("\nSaved plots to:")
print(plots_dir)


# %%
# =============================================================================
# FINAL COMPACT REPORT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

print(f"Dataset: {dataset}")
print(f"Categorical columns: {categorical_columns}")
print(f"Numerical columns: {real_value_columns}")
print(f"Train graphs: {len(train_df)}")
print(f"Valid graphs: {len(valid_df)}")
print(f"Test graphs : {len(test_df)}")
print(f"Feature dimension: {train_df['num_features'].iloc[0]}")
print(f"Number of y classes in train: {train_df['y'].nunique()}")
print(f"Number of y classes overall : {all_df['y'].nunique()}")

print("\nAverage nodes per graph:")
print(f"Train: {train_df['num_nodes'].mean():.2f}")
print(f"Valid: {valid_df['num_nodes'].mean():.2f}")
print(f"Test : {test_df['num_nodes'].mean():.2f}")

print("\nAverage edges per graph:")
print(f"Train: {train_df['num_edges'].mean():.2f}")
print(f"Valid: {valid_df['num_edges'].mean():.2f}")
print(f"Test : {test_df['num_edges'].mean():.2f}")

print("\nMost frequent y classes in train:")
print(train_df["y"].value_counts().head(10))

# %%
# %%
# =============================================================================
# SAVE FINAL REPORT AS CSV (INSIDE summary_dir)
# =============================================================================

report = {
    "dataset": dataset,

    "num_categorical_columns": len(categorical_columns),
    "num_numerical_columns": len(real_value_columns),

    "train_graphs": len(train_df),
    "valid_graphs": len(valid_df),
    "test_graphs": len(test_df),

    "feature_dimension": int(train_df["num_features"].iloc[0]),

    "num_y_classes_train": int(train_df["y"].nunique()),
    "num_y_classes_total": int(all_df["y"].nunique()),

    "avg_nodes_train": float(train_df["num_nodes"].mean()),
    "avg_nodes_valid": float(valid_df["num_nodes"].mean()),
    "avg_nodes_test": float(test_df["num_nodes"].mean()),

    "avg_edges_train": float(train_df["num_edges"].mean()),
    "avg_edges_valid": float(valid_df["num_edges"].mean()),
    "avg_edges_test": float(test_df["num_edges"].mean()),
}

report_df = pd.DataFrame([report])

# Save INSIDE same folder as other EDA tables
report_path = os.path.join(summary_dir, "final_report.csv")

# Option 1 (recommended): overwrite per dataset
report_df.to_csv(report_path, index=False)

print("\nFinal report saved to:")
print(report_path)


# Additional imports for original data EDA
import matplotlib.pyplot as plt


# %%
# =============================================================================
# PATHS
# =============================================================================

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


# %%
# =============================================================================
# CONFIG
# =============================================================================

dataset = "BPI20_RequestForPayment"
#dataset = "bpi_2012"
#dataset = "bpi_2013"
#dataset = "sp2020"

EXTRA_CATEGORICAL_COLUMNS = []
EXTRA_NUMERICAL_COLUMNS = []


# %%
# =============================================================================
# LOAD DATASET CONFIG
# =============================================================================
with open(os.path.join(root_path, "data", "dataset_features.json"), "r") as file:
    datasets_info = json.load(file)

dataset_info = datasets_info[dataset]

categorical_columns = list(dataset_info["categorical"]) + EXTRA_CATEGORICAL_COLUMNS
real_value_columns = list(dataset_info["numerical"]) + EXTRA_NUMERICAL_COLUMNS

print("\nCategorical columns:")
print(categorical_columns)
print("\nNumerical columns:")
print(real_value_columns)


# %%
# =============================================================================
# LOAD DATA
# =============================================================================

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


# %%
# =============================================================================
# TYPE CLEANING
# =============================================================================

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

tab_all["CaseID"] = tab_all["CaseID"].astype(np.str_)
tab_train["CaseID"] = tab_train["CaseID"].astype(np.str_)
tab_valid["CaseID"] = tab_valid["CaseID"].astype(np.str_)
tab_test["CaseID"] = tab_test["CaseID"].astype(np.str_)

if "time:timestamp" in tab_all.columns:
    tab_all["time:timestamp"] = pd.to_datetime(tab_all["time:timestamp"], errors="coerce")
    tab_train["time:timestamp"] = pd.to_datetime(tab_train["time:timestamp"], errors="coerce")
    tab_valid["time:timestamp"] = pd.to_datetime(tab_valid["time:timestamp"], errors="coerce")
    tab_test["time:timestamp"] = pd.to_datetime(tab_test["time:timestamp"], errors="coerce")


# %%
# =============================================================================
# OUTPUT DIR
# =============================================================================

original_eda_dir = os.path.join(results_root_dir, dataset, "eda_original_data")
os.makedirs(original_eda_dir, exist_ok=True)

print("\nOriginal data EDA output dir:")
print(original_eda_dir)


# %%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def summarize_event_log(df, split_name):
    summary = {
        "split": split_name,
        "rows": len(df),
        "num_cases": df["CaseID"].nunique(),
        "num_columns": df.shape[1],
        "num_activities": df["Activity"].nunique() if "Activity" in df.columns else np.nan,
    }

    case_lengths = df.groupby("CaseID").size()
    summary["avg_events_per_case"] = float(case_lengths.mean())
    summary["median_events_per_case"] = float(case_lengths.median())
    summary["min_events_per_case"] = int(case_lengths.min())
    summary["max_events_per_case"] = int(case_lengths.max())

    if "time:timestamp" in df.columns:
        summary["min_timestamp"] = df["time:timestamp"].min()
        summary["max_timestamp"] = df["time:timestamp"].max()
    else:
        summary["min_timestamp"] = np.nan
        summary["max_timestamp"] = np.nan

    return pd.DataFrame([summary]), case_lengths


def missing_values_table(df, split_name):
    miss = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean().values * 100)
    })
    miss["split"] = split_name
    miss = miss.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)
    return miss


def activity_distribution(df, split_name):
    if "Activity" not in df.columns:
        return pd.DataFrame(columns=["Activity", "count", "proportion", "split"])

    dist = df["Activity"].fillna("__MISSING__").value_counts(dropna=False).reset_index()
    dist.columns = ["Activity", "count"]
    dist["proportion"] = dist["count"] / dist["count"].sum()
    dist["split"] = split_name
    return dist


def numerical_summary(df, split_name, numerical_cols):
    if len(numerical_cols) == 0:
        return pd.DataFrame()

    desc = df[numerical_cols].describe().T.reset_index()
    desc = desc.rename(columns={"index": "variable"})
    desc["split"] = split_name
    return desc


def categorical_cardinality(df, split_name, categorical_cols):
    rows = []
    for col in categorical_cols:
        rows.append({
            "split": split_name,
            "column": col,
            "n_unique": df[col].nunique(dropna=True),
            "missing_count": df[col].isna().sum(),
            "top_value": df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else np.nan,
            "top_value_count": df[col].value_counts(dropna=True).iloc[0] if not df[col].value_counts(dropna=True).empty else np.nan
        })
    return pd.DataFrame(rows)


def plot_case_length_hist(case_lengths, split_name, save_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(case_lengths.values, bins=30)
    plt.title(f"Case length distribution - {split_name} - {dataset}")
    plt.xlabel("Events per case")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{split_name}_case_length_hist.png"), dpi=200, bbox_inches="tight")
    plt.show()


def plot_top_activities(activity_df, split_name, save_dir, top_n=20):
    if activity_df.empty:
        return

    top_df = activity_df.head(top_n).copy()

    plt.figure(figsize=(10, 5))
    plt.bar(top_df["Activity"].astype(str), top_df["count"])
    plt.title(f"Top {top_n} activities - {split_name} - {dataset}")
    plt.xlabel("Activity")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{split_name}_top_{top_n}_activities.png"), dpi=200, bbox_inches="tight")
    plt.show()


# %%
# =============================================================================
# SUMMARIES BY SPLIT
# =============================================================================

summary_all, case_lengths_all = summarize_event_log(tab_all, "all")
summary_train, case_lengths_train = summarize_event_log(tab_train, "train")
summary_valid, case_lengths_valid = summarize_event_log(tab_valid, "valid")
summary_test, case_lengths_test = summarize_event_log(tab_test, "test")

summary_df = pd.concat(
    [summary_all, summary_train, summary_valid, summary_test],
    ignore_index=True
)

print("\n" + "=" * 80)
print("EVENT LOG SUMMARY")
print("=" * 80)
print(summary_df)


# %%
# =============================================================================
# MISSING VALUES
# =============================================================================

missing_all = missing_values_table(tab_all, "all")
missing_train = missing_values_table(tab_train, "train")
missing_valid = missing_values_table(tab_valid, "valid")
missing_test = missing_values_table(tab_test, "test")

missing_df = pd.concat(
    [missing_all, missing_train, missing_valid, missing_test],
    ignore_index=True
)

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
print(missing_df.head(30))


# %%
# =============================================================================
# ACTIVITY DISTRIBUTION
# =============================================================================

activity_all = activity_distribution(tab_all, "all")
activity_train = activity_distribution(tab_train, "train")
activity_valid = activity_distribution(tab_valid, "valid")
activity_test = activity_distribution(tab_test, "test")

activity_df = pd.concat(
    [activity_all, activity_train, activity_valid, activity_test],
    ignore_index=True
)

print("\n" + "=" * 80)
print("ACTIVITY DISTRIBUTION")
print("=" * 80)
print(activity_train.head(20))


# %%
# =============================================================================
# NUMERICAL AND CATEGORICAL SUMMARIES
# =============================================================================

numerical_all = numerical_summary(tab_all, "all", real_value_columns)
numerical_train = numerical_summary(tab_train, "train", real_value_columns)
numerical_valid = numerical_summary(tab_valid, "valid", real_value_columns)
numerical_test = numerical_summary(tab_test, "test", real_value_columns)

numerical_df = pd.concat(
    [numerical_all, numerical_train, numerical_valid, numerical_test],
    ignore_index=True
) if len(real_value_columns) > 0 else pd.DataFrame()

categorical_all = categorical_cardinality(tab_all, "all", categorical_columns)
categorical_train = categorical_cardinality(tab_train, "train", categorical_columns)
categorical_valid = categorical_cardinality(tab_valid, "valid", categorical_columns)
categorical_test = categorical_cardinality(tab_test, "test", categorical_columns)

categorical_df = pd.concat(
    [categorical_all, categorical_train, categorical_valid, categorical_test],
    ignore_index=True
)

print("\n" + "=" * 80)
print("CATEGORICAL CARDINALITY")
print("=" * 80)
print(categorical_df)

if not numerical_df.empty:
    print("\n" + "=" * 80)
    print("NUMERICAL SUMMARY")
    print("=" * 80)
    print(numerical_df)


# %%
# =============================================================================
# CLASS SPACE CHECK
# =============================================================================

if "Activity" in tab_all.columns:
    train_activities = set(tab_train["Activity"].dropna().unique())
    valid_activities = set(tab_valid["Activity"].dropna().unique())
    test_activities = set(tab_test["Activity"].dropna().unique())
    all_activities = set(tab_all["Activity"].dropna().unique())

    class_space_check = pd.DataFrame({
        "split": ["train", "valid", "test", "all"],
        "num_unique_activities": [
            len(train_activities),
            len(valid_activities),
            len(test_activities),
            len(all_activities)
        ]
    })

    unseen_valid = sorted(valid_activities - train_activities)
    unseen_test = sorted(test_activities - train_activities)

    print("\n" + "=" * 80)
    print("ACTIVITY CLASS SPACE CHECK")
    print("=" * 80)
    print(class_space_check)
    print("\nActivities in valid but not in train:")
    print(unseen_valid)
    print("\nActivities in test but not in train:")
    print(unseen_test)
else:
    class_space_check = pd.DataFrame()
    unseen_valid = []
    unseen_test = []


# %%
# =============================================================================
# PLOTS
# =============================================================================

plot_case_length_hist(case_lengths_train, "train", original_eda_dir)
plot_case_length_hist(case_lengths_valid, "valid", original_eda_dir)
plot_case_length_hist(case_lengths_test, "test", original_eda_dir)

plot_top_activities(activity_train, "train", original_eda_dir, top_n=20)
plot_top_activities(activity_valid, "valid", original_eda_dir, top_n=20)
plot_top_activities(activity_test, "test", original_eda_dir, top_n=20)


# %%
# =============================================================================
# SAVE TABLES
# =============================================================================

summary_df.to_csv(os.path.join(original_eda_dir, "event_log_summary.csv"), index=False)
missing_df.to_csv(os.path.join(original_eda_dir, "missing_values.csv"), index=False)
activity_df.to_csv(os.path.join(original_eda_dir, "activity_distribution.csv"), index=False)
categorical_df.to_csv(os.path.join(original_eda_dir, "categorical_cardinality.csv"), index=False)

if not numerical_df.empty:
    numerical_df.to_csv(os.path.join(original_eda_dir, "numerical_summary.csv"), index=False)

if not class_space_check.empty:
    class_space_check.to_csv(os.path.join(original_eda_dir, "activity_class_space_check.csv"), index=False)

pd.DataFrame({"valid_not_in_train": unseen_valid}).to_csv(
    os.path.join(original_eda_dir, "valid_activities_not_in_train.csv"),
    index=False
)

pd.DataFrame({"test_not_in_train": unseen_test}).to_csv(
    os.path.join(original_eda_dir, "test_activities_not_in_train.csv"),
    index=False
)

case_lengths_train.rename("case_length").to_csv(
    os.path.join(original_eda_dir, "train_case_lengths.csv"),
    header=True
)
case_lengths_valid.rename("case_length").to_csv(
    os.path.join(original_eda_dir, "valid_case_lengths.csv"),
    header=True
)
case_lengths_test.rename("case_length").to_csv(
    os.path.join(original_eda_dir, "test_case_lengths.csv"),
    header=True
)

print("\nSaved original data EDA outputs to:")
print(original_eda_dir)


# %%
# =============================================================================
# FINAL COMPACT REPORT
# =============================================================================

final_report = {
    "dataset": dataset,
    "rows_all": len(tab_all),
    "rows_train": len(tab_train),
    "rows_valid": len(tab_valid),
    "rows_test": len(tab_test),

    "cases_all": tab_all["CaseID"].nunique(),
    "cases_train": tab_train["CaseID"].nunique(),
    "cases_valid": tab_valid["CaseID"].nunique(),
    "cases_test": tab_test["CaseID"].nunique(),

    "num_columns": tab_all.shape[1],
    "num_categorical_columns": len(categorical_columns),
    "num_numerical_columns": len(real_value_columns),

    "num_activity_classes_all": tab_all["Activity"].nunique() if "Activity" in tab_all.columns else np.nan,
    "num_activity_classes_train": tab_train["Activity"].nunique() if "Activity" in tab_train.columns else np.nan,

    "avg_events_per_case_all": float(tab_all.groupby("CaseID").size().mean()),
    "avg_events_per_case_train": float(tab_train.groupby("CaseID").size().mean()),
    "avg_events_per_case_valid": float(tab_valid.groupby("CaseID").size().mean()),
    "avg_events_per_case_test": float(tab_test.groupby("CaseID").size().mean()),

    "median_events_per_case_all": float(tab_all.groupby("CaseID").size().median()),
    "median_events_per_case_train": float(tab_train.groupby("CaseID").size().median()),
    "median_events_per_case_valid": float(tab_valid.groupby("CaseID").size().median()),
    "median_events_per_case_test": float(tab_test.groupby("CaseID").size().median()),
}

final_report_df = pd.DataFrame([final_report])

print("\n" + "=" * 80)
print("FINAL ORIGINAL DATA REPORT")
print("=" * 80)
print(final_report_df.T)

final_report_df.to_csv(os.path.join(original_eda_dir, "final_original_data_report.csv"), index=False)

print("\nFinal original data report saved to:")
print(os.path.join(original_eda_dir, "final_original_data_report.csv"))
