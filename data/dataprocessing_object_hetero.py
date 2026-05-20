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
CONFIG_FILE_NAME = "object_graph_config.json"
OUTPUT_FOLDER_NAME = "object_hetero_graphs"
MISSING_VALUE_TOKEN = "__MISSING__"
UNDIRECT_TRANSFORMATION = ToUndirected()


def to_float(value: Any) -> float:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return 0.0
    return float(numeric_value)


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


def resolve_config_path(root_path: str, configured_path: str) -> str:
    path = Path(configured_path)
    if not path.is_absolute():
        path = Path(root_path) / path
    return str(path.resolve())


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


def encode_categorical_block(
    df: pd.DataFrame,
    col: str,
    encoders: dict[str, OneHotEncoder],
) -> np.ndarray:
    values = normalize_categorical_series(df[col]).values.reshape(-1, 1)
    return encoders[col].transform(values).astype(np.float32)


def encode_numeric_block(df: pd.DataFrame, col: str) -> np.ndarray:
    values = (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(0.0)
        .values.astype(np.float32)
        .reshape(-1, 1)
    )
    return values


def encode_event_features(
    trace: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    encoders: dict[str, OneHotEncoder],
) -> torch.Tensor:
    blocks = []
    for col in categorical_cols:
        blocks.append(encode_categorical_block(trace, col, encoders))
    for col in numeric_cols:
        blocks.append(encode_numeric_block(trace, col))

    if not blocks:
        return torch.empty((len(trace), 0), dtype=torch.float32)

    return torch.tensor(np.concatenate(blocks, axis=1), dtype=torch.float32)


def build_event_chain_edges(prefix_len: int) -> torch.Tensor:
    if prefix_len <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    src = list(range(prefix_len - 1))
    dst = list(range(1, prefix_len))
    return torch.tensor([src, dst], dtype=torch.long)


def collect_object_values(prefix_trace: pd.DataFrame, source_col: str) -> list[str]:
    values = [
        normalize_categorical_value(value)
        for value in prefix_trace[source_col].tolist()
    ]
    return list(dict.fromkeys(values))


def build_object_features(
    object_values: list[str],
    source_col: str,
    object_spec: dict[str, Any],
    encoders: dict[str, OneHotEncoder],
) -> torch.Tensor:
    if len(object_values) == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    if object_spec.get("encode_source_as_feature", True):
        values = np.array(object_values, dtype=object).reshape(-1, 1)
        encoded = encoders[source_col].transform(values).astype(np.float32)
        return torch.tensor(encoded, dtype=torch.float32)

    return torch.ones((len(object_values), 1), dtype=torch.float32)


def build_object_event_edges(
    prefix_trace: pd.DataFrame,
    source_col: str,
    object_index: dict[str, int],
) -> torch.Tensor:
    edges = []
    for event_idx, value in enumerate(prefix_trace[source_col].tolist()):
        object_value = normalize_categorical_value(value)
        if object_value in object_index:
            edges.append((object_index[object_value], event_idx))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).T.contiguous()


def load_resource_handover_edges(
    root_path: str,
    graph_view: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if not graph_view.get("include_resource_handover_edges", False):
        return {}

    handover_config = graph_view["resource_handover"]
    source_path = resolve_config_path(root_path, handover_config["source_path"])
    handover_df = pd.read_csv(source_path)

    required_cols = {
        handover_config["source_resource_col"],
        handover_config["target_resource_col"],
    }
    required_cols.update(handover_config.get("edge_attributes", []))
    missing_cols = sorted(required_cols - set(handover_df.columns))
    if missing_cols:
        raise ValueError(
            f"Resource handover file is missing required columns: {missing_cols}. "
            f"File: {source_path}"
        )

    handover_edges: dict[str, list[dict[str, Any]]] = {}
    source_col = handover_config["source_resource_col"]
    target_col = handover_config["target_resource_col"]
    edge_attributes = handover_config.get("edge_attributes", [])

    for row in handover_df.to_dict("records"):
        source_resource = normalize_categorical_value(row[source_col])
        target_resource = normalize_categorical_value(row[target_col])
        edge_values = {
            attr: to_float(row[attr])
            for attr in edge_attributes
        }
        handover_edges.setdefault(source_resource, []).append(
            {
                "target_resource": target_resource,
                "edge_values": edge_values,
            }
        )

    return handover_edges


def build_resource_handover_edges(
    object_index: dict[str, int],
    handover_edges: dict[str, list[dict[str, Any]]],
    edge_attributes: list[str],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    edges = []
    edge_attr_rows = []

    for source_resource, source_idx in object_index.items():
        for handover_edge in handover_edges.get(source_resource, []):
            target_resource = handover_edge["target_resource"]
            if target_resource not in object_index:
                continue

            edges.append((source_idx, object_index[target_resource]))
            edge_attr_rows.append(
                [
                    handover_edge["edge_values"].get(attribute, 0.0)
                    for attribute in edge_attributes
                ]
            )

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        if edge_attributes:
            edge_attr = torch.empty((0, len(edge_attributes)), dtype=torch.float32)
        else:
            edge_attr = None
        return edge_index, edge_attr

    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    if not edge_attributes:
        return edge_index, None

    edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32)
    return edge_index, edge_attr


def get_next_activity_label(
    trace: pd.DataFrame,
    next_event_idx: int,
    label_col: str,
    label_to_index: dict[str, int],
) -> torch.Tensor:
    next_label = normalize_categorical_value(trace.iloc[next_event_idx][label_col])
    class_idx = label_to_index[next_label]
    return torch.tensor([class_idx], dtype=torch.long)


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


def build_prefix_object_graphs_from_trace(
    trace: pd.DataFrame,
    dataset_name: str,
    config: dict[str, Any],
    graph_view_name: str,
    graph_view: dict[str, Any],
    resource_handover_edges: dict[str, list[dict[str, Any]]],
    encoders: dict[str, OneHotEncoder],
    label_maps: dict[str, dict[str, int]],
) -> list[HeteroData]:
    graphs = []
    timestamp_col = config["timestamp_col"]
    activity_col = config["activity_col"]
    target_col = config["target_col"]
    min_prefix_length = int(config.get("min_prefix_length", 2))
    event_categorical_cols = config["event_features"].get("categorical", [])
    event_numeric_cols = config["event_features"].get("numeric", [])
    object_nodes = config.get("object_nodes", {})
    handover_config = graph_view.get("resource_handover", {})
    include_handover_edges = graph_view.get("include_resource_handover_edges", False)

    trace = add_relative_timestamp(trace, timestamp_col)
    max_prefix_len = len(trace) - 1

    for prefix_len in range(min_prefix_length, max_prefix_len + 1):
        next_event_idx = prefix_len
        prefix_trace = trace.iloc[:prefix_len].reset_index(drop=True)

        graph = HeteroData()
        graph["event"].x = encode_event_features(
            prefix_trace,
            event_categorical_cols,
            event_numeric_cols,
            encoders,
        )
        graph["event", "follows", "event"].edge_index = build_event_chain_edges(prefix_len)

        if target_col == "next_activity":
            y_activity = get_next_activity_label(
                trace,
                next_event_idx,
                activity_col,
                label_maps[activity_col],
            )
        else:
            y_activity = get_next_activity_label(
                trace,
                next_event_idx,
                target_col,
                label_maps[target_col],
            )

        graph.y = y_activity
        graph.y_activity = y_activity
        if "org:resource" in trace.columns and "org:resource" in label_maps:
            graph.y_resource = get_next_activity_label(
                trace,
                next_event_idx,
                "org:resource",
                label_maps["org:resource"],
            )
        if timestamp_col in trace.columns:
            graph.y_time = get_next_numeric_label(
                trace,
                next_event_idx,
                timestamp_col,
            )
        graph.prefix_len = prefix_len
        graph.case_len = len(trace)
        graph.dataset_name = dataset_name
        graph.graph_view = graph_view_name
        graph.target_col = target_col
        graph.num_event_node_features_ = graph["event"].x.shape[1]

        for object_name, object_spec in object_nodes.items():
            source_col = object_spec["source_col"]
            node_type = object_spec.get("node_type", object_name)
            edge_type = object_spec.get("edge_type", "interacts")

            object_values = collect_object_values(prefix_trace, source_col)
            object_index = {value: idx for idx, value in enumerate(object_values)}

            graph[node_type].x = build_object_features(
                object_values,
                source_col,
                object_spec,
                encoders,
            )
            graph[node_type, edge_type, "event"].edge_index = build_object_event_edges(
                prefix_trace,
                source_col,
                object_index,
            )

        graph = UNDIRECT_TRANSFORMATION(graph)
        graph.is_undirected_graph = not include_handover_edges

        if include_handover_edges:
            node_type = handover_config.get("node_type", "resource")
            edge_type = handover_config.get("edge_type", "handover")
            edge_attributes = handover_config.get("edge_attributes", [])
            weight_attribute = handover_config.get("weight_attribute")
            object_spec = next(
                (
                    spec
                    for object_name, spec in object_nodes.items()
                    if spec.get("node_type", object_name) == node_type
                ),
                None,
            )
            if object_spec is None:
                raise ValueError(
                    f"Graph view '{graph_view_name}' expects handover node type "
                    f"'{node_type}', but no object node with that node_type exists."
                )

            object_values = collect_object_values(prefix_trace, object_spec["source_col"])
            object_index = {value: idx for idx, value in enumerate(object_values)}
            edge_index, edge_attr = build_resource_handover_edges(
                object_index,
                resource_handover_edges,
                edge_attributes,
            )
            graph[node_type, edge_type, node_type].edge_index = edge_index
            if edge_attr is not None:
                graph[node_type, edge_type, node_type].edge_attr = edge_attr
                if weight_attribute in edge_attributes:
                    weight_index = edge_attributes.index(weight_attribute)
                    graph[node_type, edge_type, node_type].edge_weight = (
                        edge_attr[:, weight_index]
                    )
        graphs.append(graph)

    return graphs


def get_case_ids(df: pd.DataFrame, case_id_col: str) -> list[str]:
    return list(df[case_id_col].astype(np.str_).unique())


def build_split_object_graphs(
    df_split: pd.DataFrame,
    split_name: str,
    dataset_name: str,
    config: dict[str, Any],
    graph_view_name: str,
    graph_view: dict[str, Any],
    resource_handover_edges: dict[str, list[dict[str, Any]]],
    encoders: dict[str, OneHotEncoder],
    label_maps: dict[str, dict[str, int]],
) -> list[HeteroData]:
    print(f"\nPreparing {split_name} object-centric heterogeneous dataset...")
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
            graphs = build_prefix_object_graphs_from_trace(
                trace,
                dataset_name,
                config,
                graph_view_name,
                graph_view,
                resource_handover_edges,
                encoders,
                label_maps,
            )
            split_graphs.extend(graphs)

    print(f"{split_name} object graphs created: {len(split_graphs)}")
    return split_graphs


def load_processed_splits(processed_dir: str, dataset_name: str) -> dict[str, pd.DataFrame]:
    dataset_dir = os.path.join(processed_dir, dataset_name)
    return {
        "train": pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}_processed_train.csv")),
        "validation": pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}_processed_valid.csv")),
        "test": pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}_processed_test.csv")),
        "all": pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}_processed_all.csv")),
    }


def get_processed_split_paths(processed_dir: str, dataset_name: str) -> dict[str, str]:
    dataset_dir = os.path.join(processed_dir, dataset_name)
    return {
        "train": os.path.join(dataset_dir, f"{dataset_name}_processed_train.csv"),
        "validation": os.path.join(dataset_dir, f"{dataset_name}_processed_valid.csv"),
        "test": os.path.join(dataset_dir, f"{dataset_name}_processed_test.csv"),
        "all": os.path.join(dataset_dir, f"{dataset_name}_processed_all.csv"),
    }


def validate_columns(df: pd.DataFrame, dataset_name: str, config: dict[str, Any]) -> None:
    required_cols = {
        config["case_id_col"],
        config["timestamp_col"],
        config["activity_col"],
    }
    required_cols.update(config["event_features"].get("categorical", []))
    required_cols.update(config["event_features"].get("numeric", []))
    for object_spec in config.get("object_nodes", {}).values():
        required_cols.add(object_spec["source_col"])

    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {missing_cols}"
        )


def get_encoder_columns(config: dict[str, Any]) -> list[str]:
    cols = list(config["event_features"].get("categorical", []))
    for object_spec in config.get("object_nodes", {}).values():
        source_col = object_spec["source_col"]
        if object_spec.get("encode_source_as_feature", True):
            cols.append(source_col)
    return list(dict.fromkeys(cols))


def save_object_metadata(
    save_dir: str,
    root_path: str,
    processed_dir: str,
    graph_dir: str,
    config_path: str,
    dataset_name: str,
    config: dict[str, Any],
    graph_view_name: str,
    graph_view: dict[str, Any],
    encoders: dict[str, OneHotEncoder],
    activity_classes: list[str],
    resource_classes: list[str],
    split_paths: dict[str, str],
    output_files: list[str],
) -> None:
    handover_config = graph_view.get("resource_handover", {})
    include_handover_edges = graph_view.get("include_resource_handover_edges", False)
    resolved_handover_path = None
    if include_handover_edges:
        resolved_handover_path = resolve_config_path(
            root_path,
            handover_config["source_path"],
        )

    metadata = {
        "dataset": dataset_name,
        "graph_view": graph_view_name,
        "graph_type": "object_heterogeneous",
        "is_undirected": not include_handover_edges,
        "directionality_note": (
            "event/object edges are made undirected with ToUndirected; "
            "resource handover edges remain directed."
            if include_handover_edges
            else "all generated edges are made undirected with ToUndirected."
        ),
        "paths": {
            "project_root": root_path,
            "processed_dir": processed_dir,
            "graph_dir": graph_dir,
            "save_dir": save_dir,
            "config_path": config_path,
            "input_splits": split_paths,
            "output_files": output_files,
            "resource_handover_source": resolved_handover_path,
        },
        "case_id_col": config["case_id_col"],
        "timestamp_col": config["timestamp_col"],
        "activity_col": config["activity_col"],
        "target_col": config["target_col"],
        "min_prefix_length": int(config.get("min_prefix_length", 2)),
        "event_features": config.get("event_features", {}),
        "object_nodes": config.get("object_nodes", {}),
        "node_types": ["event"]
        + [
            spec.get("node_type", object_name)
            for object_name, spec in config.get("object_nodes", {}).items()
        ],
        "edge_types": [["event", "follows", "event"]]
        + [
            [
                spec.get("node_type", object_name),
                spec.get("edge_type", "interacts"),
                "event",
            ]
            for object_name, spec in config.get("object_nodes", {}).items()
        ]
        + (
            [
                [
                    handover_config.get("node_type"),
                    handover_config.get("edge_type"),
                    handover_config.get("node_type"),
                ]
            ]
            if include_handover_edges
            else []
        ),
        "reverse_edge_types": [
            [
                "event",
                f"rev_{spec.get('edge_type', 'interacts')}",
                spec.get("node_type", object_name),
            ]
            for object_name, spec in config.get("object_nodes", {}).items()
        ],
        "resource_handover_edges": include_handover_edges,
        "resource_handover": {
            "scope": "prefix_resource_nodes_only",
            "source_path": resolved_handover_path,
            "node_type": handover_config.get("node_type"),
            "edge_type": handover_config.get("edge_type"),
            "source_resource_col": handover_config.get("source_resource_col"),
            "target_resource_col": handover_config.get("target_resource_col"),
            "weight_attribute": handover_config.get("weight_attribute"),
            "edge_attributes": handover_config.get("edge_attributes", []),
        }
        if include_handover_edges
        else None,
        "target": config["target_col"],
        "activity_classes": activity_classes,
        "resource_classes": resource_classes,
        "one_hot_categories": {
            col: [str(value) for value in encoder.categories_[0]]
            for col, encoder in encoders.items()
        },
        "config": config,
        "selected_graph_view_config": graph_view,
    }

    with open(os.path.join(save_dir, "object_hetero_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build object-centric heterogeneous prefix graphs from processed event logs."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=os.environ.get("THESIS_DATASET", DEFAULT_DATASET),
        help="Dataset key defined in data/object_graph_config.json.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip graph creation when all output split files already exist.",
    )
    parser.add_argument(
        "--graph-view",
        default=os.environ.get("THESIS_GRAPH_VIEW"),
        help=(
            "Graph view key from graph_views in data/object_graph_config.json. "
            "Defaults to active_graph_view in the dataset config."
        ),
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
        "THESIS_OBJECT_GRAPHS_DIR",
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
    graph_views = config.get("graph_views", {})
    graph_view_name = args.graph_view or config.get("active_graph_view")
    if graph_view_name is None:
        graph_view_name = "default"
        graph_view = {"include_resource_handover_edges": False}
    else:
        if graph_view_name not in graph_views:
            available = ", ".join(sorted(graph_views))
            raise ValueError(
                f"Unknown graph view '{graph_view_name}' for dataset "
                f"'{dataset_name}'. Choose one of: {available}"
            )
        graph_view = graph_views[graph_view_name]

    save_dir = os.path.join(graph_dir, dataset_name, graph_view_name)
    os.makedirs(save_dir, exist_ok=True)

    output_files = [
        os.path.join(save_dir, "train_set_object_hetero.pt"),
        os.path.join(save_dir, "validation_set_object_hetero.pt"),
        os.path.join(save_dir, "test_set_object_hetero.pt"),
    ]
    if args.skip_existing and all(os.path.exists(path) for path in output_files):
        print("All object graph files already exist. Skipping creation.")
        return

    print(root_path, processed_dir, graph_dir, sep="\n")
    print(f"Selected dataset: {dataset_name}")
    print(f"Selected graph view: {graph_view_name}")
    print(f"Object graph configuration: {config_path}")

    split_paths = get_processed_split_paths(processed_dir, dataset_name)
    splits = load_processed_splits(processed_dir, dataset_name)
    validate_columns(splits["all"], dataset_name, config)
    resource_handover_edges = load_resource_handover_edges(root_path, graph_view)

    encoder_columns = get_encoder_columns(config)
    encoders = fit_one_hot_encoders(splits["all"], encoder_columns)

    activity_col = config["activity_col"]
    activity_values = normalize_categorical_series(splits["all"][activity_col])
    activity_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    activity_encoder.fit(activity_values.values.reshape(-1, 1))
    activity_classes = [str(value) for value in activity_encoder.categories_[0]]
    activity_to_index = {class_name: idx for idx, class_name in enumerate(activity_classes)}
    label_maps = {activity_col: activity_to_index}

    resource_classes = []
    if "org:resource" in encoders:
        resource_classes = [str(value) for value in encoders["org:resource"].categories_[0]]
        label_maps["org:resource"] = {
            class_name: idx for idx, class_name in enumerate(resource_classes)
        }

    train_graphs = build_split_object_graphs(
        splits["train"],
        "training",
        dataset_name,
        config,
        graph_view_name,
        graph_view,
        resource_handover_edges,
        encoders,
        label_maps,
    )
    validation_graphs = build_split_object_graphs(
        splits["validation"],
        "validation",
        dataset_name,
        config,
        graph_view_name,
        graph_view,
        resource_handover_edges,
        encoders,
        label_maps,
    )
    test_graphs = build_split_object_graphs(
        splits["test"],
        "test",
        dataset_name,
        config,
        graph_view_name,
        graph_view,
        resource_handover_edges,
        encoders,
        label_maps,
    )

    torch.save(train_graphs, output_files[0])
    torch.save(validation_graphs, output_files[1])
    torch.save(test_graphs, output_files[2])
    save_object_metadata(
        save_dir,
        root_path,
        processed_dir,
        graph_dir,
        config_path,
        dataset_name,
        config,
        graph_view_name,
        graph_view,
        encoders,
        activity_classes,
        resource_classes,
        split_paths,
        output_files,
    )

    print("\nSaved object graph files:")
    for path in output_files:
        print(path)
    print(os.path.join(save_dir, "object_hetero_metadata.json"))

    if train_graphs:
        graph = train_graphs[0]
        print("\nExample object-centric heterogeneous graph:")
        print(graph)
        print("Metadata:", graph.metadata())
        print("Target y:", graph.y)


if __name__ == "__main__":
    main()
