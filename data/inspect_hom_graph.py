import json
import os
from pathlib import Path

import pandas as pd
import torch


# Pick one dataset by uncommenting it.
# DATASET = "BPI20_RequestForPayment"
DATASET = "bpi_2012"
# DATASET = "bpi_2013"
# DATASET = "sp2020"
#DATASET = "tiny_sp2020"

# tiny_sp2020 is a small copy of sp2020 graphs, so it reuses sp2020
# processed data and feature metadata for decoding.
METADATA_DATASET = "sp2020" if DATASET == "tiny_sp2020" else DATASET

# Pick the split and graph position inside that split.
SPLIT = "train"  # "train", "validation", or "test"
GRAPH_INDEX = 0

# Increase these if you want more rows printed.
MAX_NODES_TO_PRINT = 20
MAX_EDGES_TO_PRINT = 40
MAX_ACTIVE_CATEGORIES_TO_PRINT = 10


def resolve_project_root() -> Path:
    env_root = os.environ.get("THESIS_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "data" / "dataset_features.json").exists():
            return root
        raise FileNotFoundError(
            "THESIS_PROJECT_ROOT is set, but data/dataset_features.json was "
            f"not found under: {root}"
        )

    script_path = Path(__file__).resolve()
    candidates = [
        script_path.parent.parent,
        Path.cwd().resolve(),
        Path.cwd().resolve() / "Thesis code",
        Path.cwd().resolve().parent,
    ]

    for candidate in candidates:
        if (candidate / "data" / "dataset_features.json").exists():
            return candidate

    raise FileNotFoundError("Could not locate data/dataset_features.json.")


def graph_file_name(split: str) -> str:
    names = {
        "train": "train_set_homo.pt",
        "validation": "validation_set_homo.pt",
        "valid": "validation_set_homo.pt",
        "val": "validation_set_homo.pt",
        "test": "test_set_homo.pt",
    }
    try:
        return names[split.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown split: {split}") from exc


def load_graphs(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("object").where(series.notna(), "__MISSING__")
    return normalized.astype(str)


def build_feature_layout(root: Path, metadata_dataset: str):
    features_path = root / "data" / "dataset_features.json"
    processed_path = (
        root
        / "data"
        / "datasets"
        / "processed"
        / metadata_dataset
        / f"{metadata_dataset}_processed_all.csv"
    )

    with features_path.open("r", encoding="utf-8") as file:
        dataset_info = json.load(file)[metadata_dataset]

    tab_all = pd.read_csv(processed_path)
    categorical_columns = [
        col for col in dataset_info["categorical"] if col in tab_all.columns
    ]
    numerical_columns = [
        col for col in dataset_info["numerical"] if col in tab_all.columns
    ]

    layout = []
    start = 0

    for col in categorical_columns:
        classes = sorted(normalize_categorical_series(tab_all[col]).unique())
        end = start + len(classes)
        layout.append(
            {
                "name": col,
                "kind": "categorical",
                "start": start,
                "end": end,
                "classes": classes,
            }
        )
        start = end

    for col in numerical_columns:
        end = start + 1
        layout.append(
            {
                "name": col,
                "kind": "numerical",
                "start": start,
                "end": end,
                "classes": None,
            }
        )
        start = end

    return layout


def as_python_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def decode_node_features(row, layout):
    decoded = []
    for block in layout:
        values = row[block["start"] : block["end"]]

        if block["kind"] == "categorical":
            active_positions = torch.nonzero(values > 0, as_tuple=False).flatten().tolist()
            active_values = [
                block["classes"][position]
                for position in active_positions[:MAX_ACTIVE_CATEGORIES_TO_PRINT]
            ]
            decoded.append((block["name"], active_values))
        else:
            decoded.append((block["name"], round(float(values[0]), 6)))

    return decoded


def print_header(title: str):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def print_graph_summary(graph, graph_index: int, graph_count: int, graph_path: Path):
    print_header("Graph selection")
    print(f"dataset: {DATASET}")
    print(f"metadata_dataset: {METADATA_DATASET}")
    print(f"split: {SPLIT}")
    print(f"graph_index: {graph_index} / {graph_count - 1}")
    print(f"path: {graph_path}")

    print_header("Graph object")
    print(graph)
    print(f"num_nodes: {graph.num_nodes}")
    print(f"num_edges: {graph.num_edges}")
    print(f"x shape: {tuple(graph.x.shape)}")
    print(f"edge_index shape: {tuple(graph.edge_index.shape)}")
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        print(f"edge_attr shape: {tuple(graph.edge_attr.shape)}")

    for attr in ["prefix_len", "case_len", "num_node_features_"]:
        if hasattr(graph, attr):
            print(f"{attr}: {as_python_scalar(getattr(graph, attr))}")


def print_labels(graph, layout):
    print_header("Labels")
    label_attrs = ["y", "y_activity", "y_resource", "y_time"]
    for attr in label_attrs:
        if hasattr(graph, attr):
            value = getattr(graph, attr)
            print(f"{attr}: {value}")

    activity_block = next((b for b in layout if b["name"] == "Activity"), None)
    if activity_block and hasattr(graph, "y_activity"):
        idx = int(graph.y_activity.flatten()[0])
        if idx < len(activity_block["classes"]):
            print(f"decoded y_activity: {activity_block['classes'][idx]}")

    resource_block = next((b for b in layout if b["name"] == "org:resource"), None)
    if resource_block and hasattr(graph, "y_resource"):
        idx = int(graph.y_resource.flatten()[0])
        if idx < len(resource_block["classes"]):
            print(f"decoded y_resource: {resource_block['classes'][idx]}")


def print_feature_layout(layout, graph):
    print_header("Feature layout")
    expected_width = layout[-1]["end"] if layout else 0
    print(f"expected feature width from metadata: {expected_width}")
    print(f"actual feature width in graph.x: {graph.x.shape[1]}")
    for block in layout:
        span = f"{block['start']}:{block['end']}"
        if block["kind"] == "categorical":
            print(
                f"{span} | {block['name']} | categorical | "
                f"{len(block['classes'])} classes"
            )
        else:
            print(f"{span} | {block['name']} | numerical")


def print_nodes(graph, layout):
    print_header("Nodes/events")
    n_nodes = min(graph.num_nodes, MAX_NODES_TO_PRINT)
    for node_idx in range(n_nodes):
        print(f"\nnode/event {node_idx}")
        for name, value in decode_node_features(graph.x[node_idx], layout):
            print(f"  {name}: {value}")

    if graph.num_nodes > n_nodes:
        print(f"\n... {graph.num_nodes - n_nodes} more nodes not printed")


def print_edges(graph):
    print_header("Edges")
    edge_index = graph.edge_index.t().tolist()
    n_edges = min(len(edge_index), MAX_EDGES_TO_PRINT)
    for edge_idx, (src, dst) in enumerate(edge_index[:n_edges]):
        print(f"{edge_idx}: {src} -> {dst}")

    if len(edge_index) > n_edges:
        print(f"... {len(edge_index) - n_edges} more edges not printed")


def main():
    root = resolve_project_root()
    graph_path = (
        root
        / "data"
        / "datasets"
        / "hom_graphs"
        / DATASET
        / graph_file_name(SPLIT)
    )

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")

    graphs = load_graphs(graph_path)
    if not 0 <= GRAPH_INDEX < len(graphs):
        raise IndexError(
            f"GRAPH_INDEX must be between 0 and {len(graphs) - 1}. "
            f"Current value: {GRAPH_INDEX}"
        )

    graph = graphs[GRAPH_INDEX]
    layout = build_feature_layout(root, METADATA_DATASET)

    print_graph_summary(graph, GRAPH_INDEX, len(graphs), graph_path)
    print_labels(graph, layout)
    print_feature_layout(layout, graph)
    print_nodes(graph, layout)
    print_edges(graph)


if __name__ == "__main__":
    main()
