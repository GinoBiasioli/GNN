import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm


DEFAULT_DATASET = "bpi_2012"
CONFIG_FILE_NAME = "hetero_graph_config.json"
OUTPUT_FOLDER_NAME = "categorical_value_hetero_graphs"
MISSING_VALUE_TOKEN = "__MISSING__"
UNDIRECT_TRANSFORMATION = ToUndirected()


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


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("object").where(series.notna(), MISSING_VALUE_TOKEN)
    return normalized.astype(np.str_)


def normalize_categorical_value(value: Any) -> str:
    if pd.isna(value):
        return MISSING_VALUE_TOKEN
    return str(value)


def fit_one_hot_encoders(dataset_df: pd.DataFrame, columns: list[str]) -> dict[str, OneHotEncoder]:
    encoders = {}
    for col in columns:
        values = normalize_categorical_series(dataset_df[col]).values.reshape(-1, 1)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(values)
        encoders[col] = encoder
    return encoders


def encode_categorical_values(
    values: list[str],
    col: str,
    encoders: dict[str, OneHotEncoder],
) -> torch.Tensor:
    if not values:
        feature_dim = len(encoders[col].categories_[0])
        return torch.empty((0, feature_dim), dtype=torch.float32)

    encoded = encoders[col].transform(np.array(values, dtype=object).reshape(-1, 1))
    return torch.tensor(encoded.astype(np.float32), dtype=torch.float32)


def encode_numeric_block(df: pd.DataFrame, col: str) -> np.ndarray:
    return (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(0.0)
        .values.astype(np.float32)
        .reshape(-1, 1)
    )


def encode_event_features(
    prefix_trace: pd.DataFrame,
    activity_col: str,
    numeric_cols: list[str],
    encoders: dict[str, OneHotEncoder],
) -> torch.Tensor:
    blocks = []

    activity_values = normalize_categorical_series(prefix_trace[activity_col]).values.reshape(-1, 1)
    blocks.append(encoders[activity_col].transform(activity_values).astype(np.float32))

    for col in numeric_cols:
        blocks.append(encode_numeric_block(prefix_trace, col))

    return torch.tensor(np.concatenate(blocks, axis=1), dtype=torch.float32)


def build_event_chain_edges(prefix_len: int) -> torch.Tensor:
    if prefix_len <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    src = list(range(prefix_len - 1))
    dst = list(range(1, prefix_len))
    return torch.tensor([src, dst], dtype=torch.long)


def collect_unique_values(prefix_trace: pd.DataFrame, col: str) -> list[str]:
    values = [
        normalize_categorical_value(value)
        for value in prefix_trace[col].tolist()
    ]
    return list(dict.fromkeys(values))


def build_event_to_value_edges(
    prefix_trace: pd.DataFrame,
    col: str,
    value_index: dict[str, int],
) -> torch.Tensor:
    edges = []
    for event_idx, value in enumerate(prefix_trace[col].tolist()):
        normalized_value = normalize_categorical_value(value)
        edges.append((event_idx, value_index[normalized_value]))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).T.contiguous()


def get_next_class_label(
    trace: pd.DataFrame,
    next_event_idx: int,
    label_col: str,
    label_to_index: dict[str, int],
) -> torch.Tensor:
    label = normalize_categorical_value(trace.iloc[next_event_idx][label_col])
    return torch.tensor([label_to_index[label]], dtype=torch.long)


def get_next_numeric_label(
    trace: pd.DataFrame,
    next_event_idx: int,
    label_col: str,
) -> torch.Tensor:
    value = pd.to_numeric(
        pd.Series([trace.iloc[next_event_idx][label_col]]),
        errors="coerce",
    ).fillna(0.0).iloc[0]
    return torch.tensor([float(value)], dtype=torch.float32)


def add_relative_timestamp(trace: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    trace_copy = trace.copy()
    if timestamp_col not in trace_copy.columns or len(trace_copy) == 0:
        return trace_copy

    timestamps = pd.to_numeric(trace_copy[timestamp_col], errors="coerce").fillna(0.0)
    first_timestamp = float(timestamps.iloc[0])
    trace_copy[timestamp_col] = timestamps - first_timestamp
    return trace_copy


def build_prefix_categorical_value_graphs_from_trace(
    trace: pd.DataFrame,
    dataset_name: str,
    config: dict[str, Any],
    encoders: dict[str, OneHotEncoder],
    label_maps: dict[str, dict[str, int]],
) -> list[HeteroData]:
    graphs = []
    timestamp_col = config["timestamp_col"]
    activity_col = config["activity_col"]
    target_col = config["target_col"]
    min_prefix_length = int(config.get("min_prefix_length", 2))
    categorical_cols = [col for col in config.get("categorical", []) if col in trace.columns]
    numeric_cols = [col for col in config.get("numerical", []) if col in trace.columns]
    value_node_cols = [col for col in categorical_cols if col != activity_col]

    if activity_col not in categorical_cols:
        raise ValueError(
            f"Activity column '{activity_col}' must be included in the categorical config."
        )

    trace = add_relative_timestamp(trace, timestamp_col)
    max_prefix_len = len(trace) - 1

    for prefix_len in range(min_prefix_length, max_prefix_len + 1):
        next_event_idx = prefix_len
        prefix_trace = trace.iloc[:prefix_len].reset_index(drop=True)
        graph = HeteroData()

        graph["event"].x = encode_event_features(
            prefix_trace,
            activity_col,
            numeric_cols,
            encoders,
        )
        graph["event", "follows", "event"].edge_index = build_event_chain_edges(prefix_len)

        for col in value_node_cols:
            values = collect_unique_values(prefix_trace, col)
            value_index = {value: idx for idx, value in enumerate(values)}

            graph[col].x = encode_categorical_values(values, col, encoders)
            graph["event", f"has_{col}", col].edge_index = build_event_to_value_edges(
                prefix_trace,
                col,
                value_index,
            )

        if target_col == "next_activity":
            y_activity = get_next_class_label(
                trace,
                next_event_idx,
                activity_col,
                label_maps[activity_col],
            )
        else:
            y_activity = get_next_class_label(
                trace,
                next_event_idx,
                target_col,
                label_maps[target_col],
            )

        graph.y = y_activity
        graph.y_activity = y_activity
        if "org:resource" in trace.columns and "org:resource" in label_maps:
            graph.y_resource = get_next_class_label(
                trace,
                next_event_idx,
                "org:resource",
                label_maps["org:resource"],
            )
        if timestamp_col in trace.columns:
            graph.y_time = get_next_numeric_label(trace, next_event_idx, timestamp_col)

        graph.prefix_len = prefix_len
        graph.case_len = len(trace)
        graph.dataset_name = dataset_name
        graph.target_col = target_col
        graph.target_node_type = "event"
        graph.graph_type = "categorical_value_heterogeneous"
        graph.activity_col = activity_col
        graph.num_event_node_features_ = graph["event"].x.shape[1]

        graph = UNDIRECT_TRANSFORMATION(graph)
        graph.is_undirected_graph = True
        graphs.append(graph)

    return graphs


def get_case_ids(df: pd.DataFrame, case_id_col: str) -> list[str]:
    return list(df[case_id_col].astype(np.str_).unique())


def build_split_categorical_value_graphs(
    df_split: pd.DataFrame,
    split_name: str,
    dataset_name: str,
    config: dict[str, Any],
    encoders: dict[str, OneHotEncoder],
    label_maps: dict[str, dict[str, int]],
) -> list[HeteroData]:
    print(f"\nPreparing {split_name} categorical-value heterogeneous dataset...")
    case_id_col = config["case_id_col"]
    case_ids = get_case_ids(df_split, case_id_col)
    split_graphs = []

    for case_id in tqdm(case_ids):
        trace = (
            df_split[df_split[case_id_col].astype(np.str_) == str(case_id)]
            .reset_index(drop=True)
            .drop(columns=[case_id_col])
        )

        if len(trace) > int(config.get("min_prefix_length", 2)):
            split_graphs.extend(
                build_prefix_categorical_value_graphs_from_trace(
                    trace,
                    dataset_name,
                    config,
                    encoders,
                    label_maps,
                )
            )

    print(f"{split_name} categorical-value heterogeneous graphs created: {len(split_graphs)}")
    return split_graphs


def load_processed_splits(processed_dir: str, dataset_name: str) -> dict[str, pd.DataFrame]:
    dataset_dir = os.path.join(processed_dir, dataset_name)
    return {
        "train": pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}_processed_train.csv")),
        "validation": pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}_processed_valid.csv")),
        "test": pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}_processed_test.csv")),
        "all": pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}_processed_all.csv")),
    }


def validate_columns(df: pd.DataFrame, dataset_name: str, config: dict[str, Any]) -> None:
    required_cols = {
        config["case_id_col"],
        config["timestamp_col"],
        config["activity_col"],
    }
    required_cols.update(config.get("categorical", []))
    required_cols.update(config.get("numerical", []))

    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {missing_cols}"
        )


def get_available_config(config: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    available_config = dict(config)
    available_config["categorical"] = [
        col for col in config.get("categorical", []) if col in df.columns
    ]
    available_config["numerical"] = [
        col for col in config.get("numerical", []) if col in df.columns
    ]
    return available_config


def save_categorical_value_metadata(
    save_dir: str,
    dataset_name: str,
    config: dict[str, Any],
    encoders: dict[str, OneHotEncoder],
    activity_classes: list[str],
    resource_classes: list[str],
) -> None:
    activity_col = config["activity_col"]
    categorical_cols = list(config.get("categorical", []))
    numeric_cols = list(config.get("numerical", []))
    value_node_cols = [col for col in categorical_cols if col != activity_col]
    node_types = ["event"] + value_node_cols
    edge_types = [["event", "follows", "event"]]
    edge_types.extend(
        [["event", f"has_{col}", col] for col in value_node_cols]
    )

    metadata = {
        "dataset": dataset_name,
        "graph_type": "categorical_value_heterogeneous",
        "is_undirected": True,
        "node_types": node_types,
        "edge_types": edge_types,
        "target": config["target_col"],
        "target_node_type": "event",
        "activity_col": activity_col,
        "event_features": {
            "categorical": [activity_col],
            "numerical": numeric_cols,
        },
        "value_node_columns": value_node_cols,
        "activity_classes": activity_classes,
        "resource_classes": resource_classes,
        "one_hot_categories": {
            col: [str(value) for value in encoder.categories_[0]]
            for col, encoder in encoders.items()
        },
        "config": config,
    }

    with open(os.path.join(save_dir, "categorical_value_hetero_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build categorical-value heterogeneous prefix graphs from processed event logs."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=os.environ.get("THESIS_DATASET", DEFAULT_DATASET),
        help="Dataset key defined in data/hetero_graph_config.json.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip graph creation when all output split files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = resolve_project_root()
    processed_dir = resolve_dir(
        "THESIS_PROCESSED_DIR",
        os.path.join(root_path, "data", "datasets", "processed"),
    )
    graph_dir = resolve_dir(
        "THESIS_CATEGORICAL_VALUE_HETERO_GRAPHS_DIR",
        os.path.join(root_path, "data", "datasets", OUTPUT_FOLDER_NAME),
    )
    config_path = os.path.join(root_path, "data", CONFIG_FILE_NAME)

    with open(config_path, "r") as file:
        configs = json.load(file)

    if args.dataset not in configs:
        available = ", ".join(sorted(configs))
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. Choose one of: {available}"
        )

    dataset_name = args.dataset
    config = configs[dataset_name]
    save_dir = os.path.join(graph_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    output_files = [
        os.path.join(save_dir, "train_set_categorical_value_hetero.pt"),
        os.path.join(save_dir, "validation_set_categorical_value_hetero.pt"),
        os.path.join(save_dir, "test_set_categorical_value_hetero.pt"),
    ]
    if args.skip_existing and all(os.path.exists(path) for path in output_files):
        print("All categorical-value heterogeneous graph files already exist. Skipping creation.")
        return

    print(root_path, processed_dir, graph_dir, sep="\n")
    print(f"Selected dataset: {dataset_name}")
    print(f"Categorical-value heterogeneous graph configuration: {config_path}")

    splits = load_processed_splits(processed_dir, dataset_name)
    validate_columns(splits["all"], dataset_name, config)
    config = get_available_config(config, splits["all"])

    categorical_cols = config.get("categorical", [])
    encoders = fit_one_hot_encoders(splits["all"], categorical_cols)

    activity_col = config["activity_col"]
    activity_values = normalize_categorical_series(splits["all"][activity_col])
    activity_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    activity_encoder.fit(activity_values.values.reshape(-1, 1))
    activity_classes = [str(value) for value in activity_encoder.categories_[0]]
    label_maps = {
        activity_col: {
            class_name: idx for idx, class_name in enumerate(activity_classes)
        }
    }

    resource_classes = []
    if "org:resource" in encoders:
        resource_classes = [str(value) for value in encoders["org:resource"].categories_[0]]
        label_maps["org:resource"] = {
            class_name: idx for idx, class_name in enumerate(resource_classes)
        }

    train_graphs = build_split_categorical_value_graphs(
        splits["train"],
        "training",
        dataset_name,
        config,
        encoders,
        label_maps,
    )
    validation_graphs = build_split_categorical_value_graphs(
        splits["validation"],
        "validation",
        dataset_name,
        config,
        encoders,
        label_maps,
    )
    test_graphs = build_split_categorical_value_graphs(
        splits["test"],
        "test",
        dataset_name,
        config,
        encoders,
        label_maps,
    )

    torch.save(train_graphs, output_files[0])
    torch.save(validation_graphs, output_files[1])
    torch.save(test_graphs, output_files[2])
    save_categorical_value_metadata(
        save_dir,
        dataset_name,
        config,
        encoders,
        activity_classes,
        resource_classes,
    )

    print("\nSaved categorical-value heterogeneous graph files:")
    for path in output_files:
        print(path)
    print(os.path.join(save_dir, "categorical_value_hetero_metadata.json"))

    if train_graphs:
        graph = train_graphs[0]
        print("\nExample categorical-value heterogeneous graph:")
        print(graph)
        print("Metadata:", graph.metadata())
        print("Target y:", graph.y)


if __name__ == "__main__":
    main()
