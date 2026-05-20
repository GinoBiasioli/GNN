import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


MISSING_VALUE_TOKEN = "__MISSING__"


def resolve_project_root() -> Path:
    env_root = os.environ.get("THESIS_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "data" / "dataset_features.json").exists():
            return candidate
        raise FileNotFoundError(
            "THESIS_PROJECT_ROOT is set, but data/dataset_features.json was not found "
            f"under: {candidate}"
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
            return candidate

    raise FileNotFoundError("Could not locate project root with data/dataset_features.json.")


def resolve_dir(env_var_name: str, default_path: Path) -> Path:
    env_value = os.environ.get(env_var_name)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return default_path


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return MISSING_VALUE_TOKEN
    return str(value)


def normalize_series(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING_VALUE_TOKEN).astype(str)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_processed_split(processed_dir: Path, dataset: str, split: str) -> pd.DataFrame:
    dataset_dir = processed_dir / dataset
    split_suffix = "valid" if split == "validation" else split
    csv_path = dataset_dir / f"{dataset}_processed_{split_suffix}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed split not found: {csv_path}")
    return pd.read_csv(csv_path)


def fit_category_sizes(df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, int]:
    category_sizes = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(normalize_series(df[col]).values.reshape(-1, 1))
        category_sizes[col] = len(encoder.categories_[0])
    return category_sizes


def csv_join(values: list[Any]) -> str:
    return " | ".join(str(value) for value in values)


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_case_ids(raw_case_ids: str | None, case_file: str | None) -> list[str]:
    case_ids = []
    if raw_case_ids:
        case_ids.extend([case_id.strip() for case_id in raw_case_ids.split(",") if case_id.strip()])
    if case_file:
        with open(case_file, "r", encoding="utf-8") as file:
            case_ids.extend([line.strip() for line in file if line.strip()])
    return list(dict.fromkeys(case_ids))


def parse_prefix_lengths(raw_prefix_lengths: str | None, min_prefix_len: int, max_prefix_len: int) -> list[int]:
    if max_prefix_len < min_prefix_len:
        return []
    if raw_prefix_lengths is None or raw_prefix_lengths == "all":
        return list(range(min_prefix_len, max_prefix_len + 1))
    if raw_prefix_lengths == "last":
        return [max_prefix_len]
    values = []
    for token in raw_prefix_lengths.split(","):
        token = token.strip()
        if not token:
            continue
        prefix_len = int(token)
        if min_prefix_len <= prefix_len <= max_prefix_len:
            values.append(prefix_len)
    return list(dict.fromkeys(values))


def edge_density(num_nodes: int, num_edges: int) -> float:
    if num_nodes <= 1:
        return 0.0
    return round(num_edges / (num_nodes * (num_nodes - 1)), 6)


def get_next_event_summary(
    trace: pd.DataFrame,
    next_idx: int,
    activity_col: str,
    resource_col: str = "org:resource",
    timestamp_col: str = "time:timestamp",
) -> dict[str, Any]:
    row = trace.iloc[next_idx]
    summary = {
        "next_event_index": next_idx,
        "next_activity": normalize_value(row[activity_col]) if activity_col in trace.columns else "",
    }
    if resource_col in trace.columns:
        summary["next_resource"] = normalize_value(row[resource_col])
    if timestamp_col in trace.columns:
        summary["next_timestamp"] = row[timestamp_col]
    return summary


def summarize_homo(
    dataset_info: dict[str, Any],
    trace: pd.DataFrame,
    case_id: str,
    prefix_len: int,
    activity_col: str,
    timestamp_col: str,
    category_sizes: dict[str, int],
) -> dict[str, Any]:
    categorical_cols = [col for col in dataset_info.get("categorical", []) if col in trace.columns]
    numerical_cols = [col for col in dataset_info.get("numerical", []) if col in trace.columns]
    feature_dim = sum(category_sizes.get(col, 0) for col in categorical_cols) + len(numerical_cols)
    directed_chain_edges = max(prefix_len - 1, 0)
    edge_type_counts = {"event__follows__event": directed_chain_edges * 2}
    node_type_counts = {"event": prefix_len}
    node_semantics = (
        "Each node is one prefix event. Its feature vector concatenates one-hot categorical "
        "blocks and raw numeric event columns."
    )
    edge_semantics = "Edges connect consecutive prefix events and are then duplicated in reverse."

    return {
        "case_id": case_id,
        "representation": "homo",
        "prefix_len": prefix_len,
        "case_len": len(trace),
        "num_nodes": prefix_len,
        "num_edges": sum(edge_type_counts.values()),
        "node_types": "event",
        "edge_types": "event__follows__event",
        "node_type_counts": json_compact(node_type_counts),
        "edge_type_counts": json_compact(edge_type_counts),
        "node_feature_dims": json_compact({"event": feature_dim}),
        "node_semantics": node_semantics,
        "edge_semantics": edge_semantics,
        "feature_columns": csv_join(categorical_cols + numerical_cols),
        "sample_nodes": csv_join(
            [
                f"event[{idx}]={normalize_value(trace.iloc[idx][activity_col])}"
                for idx in range(min(prefix_len, 3))
            ]
        ),
        "sample_edges": csv_join(
            [f"event[{idx}]->event[{idx + 1}]" for idx in range(min(prefix_len - 1, 3))]
        ),
        "density": edge_density(prefix_len, sum(edge_type_counts.values())),
        **get_next_event_summary(trace, prefix_len, activity_col, timestamp_col=timestamp_col),
    }


def summarize_regular_hetero(
    config: dict[str, Any],
    trace: pd.DataFrame,
    case_id: str,
    prefix_len: int,
    category_sizes: dict[str, int],
) -> dict[str, Any]:
    activity_col = config["activity_col"]
    timestamp_col = config["timestamp_col"]
    categorical_cols = [col for col in config.get("categorical", []) if col in trace.columns]
    numerical_cols = [col for col in config.get("numerical", []) if col in trace.columns]
    node_cols = list(dict.fromkeys(categorical_cols + numerical_cols))
    node_type_counts = {col: prefix_len for col in node_cols}
    node_feature_dims = {
        col: category_sizes.get(col, 1) if col in categorical_cols else 1
        for col in node_cols
    }

    edge_type_counts = {}
    for col in node_cols:
        edge_type_counts[f"{col}__follows__{col}"] = max(prefix_len - 1, 0) * 2
    for col in node_cols:
        if col == activity_col:
            continue
        edge_type_counts[f"{activity_col}__has_{col}__{col}"] = prefix_len
        edge_type_counts[f"{col}__rev_has_{col}__{activity_col}"] = prefix_len

    num_nodes = sum(node_type_counts.values())
    num_edges = sum(edge_type_counts.values())

    return {
        "case_id": case_id,
        "representation": "hetero",
        "prefix_len": prefix_len,
        "case_len": len(trace),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_types": csv_join(node_cols),
        "edge_types": csv_join(list(edge_type_counts)),
        "node_type_counts": json_compact(node_type_counts),
        "edge_type_counts": json_compact(edge_type_counts),
        "node_feature_dims": json_compact(node_feature_dims),
        "node_semantics": (
            "Each configured column is a node type; each prefix position has one node per "
            "column/node type."
        ),
        "edge_semantics": (
            "Follows connects consecutive values within each node type. Has connects the "
            "activity node at a position to the other attribute nodes at the same position; "
            "reverse edges are added."
        ),
        "feature_columns": csv_join(node_cols),
        "sample_nodes": csv_join(
            [
                f"{col}[0]={normalize_value(trace.iloc[0][col])}"
                for col in node_cols[:3]
                if prefix_len > 0
            ]
        ),
        "sample_edges": csv_join(
            [f"{col}[0]->{col}[1]" for col in node_cols[:2] if prefix_len > 1]
            + [
                f"{activity_col}[0]->{col}[0]"
                for col in node_cols
                if col != activity_col
            ][:2]
        ),
        "density": edge_density(num_nodes, num_edges),
        **get_next_event_summary(trace, prefix_len, activity_col, timestamp_col=timestamp_col),
    }


def summarize_object_hetero(
    config: dict[str, Any],
    trace: pd.DataFrame,
    case_id: str,
    prefix_len: int,
    category_sizes: dict[str, int],
) -> dict[str, Any]:
    activity_col = config["activity_col"]
    timestamp_col = config["timestamp_col"]
    event_categorical_cols = [
        col for col in config["event_features"].get("categorical", []) if col in trace.columns
    ]
    event_numeric_cols = [
        col for col in config["event_features"].get("numeric", []) if col in trace.columns
    ]
    prefix_trace = trace.iloc[:prefix_len]

    node_type_counts = {"event": prefix_len}
    node_feature_dims = {
        "event": sum(category_sizes.get(col, 0) for col in event_categorical_cols)
        + len(event_numeric_cols)
    }
    edge_type_counts = {"event__follows__event": max(prefix_len - 1, 0) * 2}
    object_descriptions = []

    for object_name, object_spec in config.get("object_nodes", {}).items():
        source_col = object_spec["source_col"]
        if source_col not in trace.columns:
            continue
        node_type = object_spec.get("node_type", object_name)
        edge_type = object_spec.get("edge_type", "interacts")
        object_values = list(dict.fromkeys(normalize_value(value) for value in prefix_trace[source_col]))
        node_type_counts[node_type] = len(object_values)
        if object_spec.get("encode_source_as_feature", True):
            node_feature_dims[node_type] = category_sizes.get(source_col, 1)
        else:
            node_feature_dims[node_type] = 1
        edge_type_counts[f"{node_type}__{edge_type}__event"] = prefix_len
        edge_type_counts[f"event__rev_{edge_type}__{node_type}"] = prefix_len
        object_descriptions.append(f"{node_type} from {source_col}")

    num_nodes = sum(node_type_counts.values())
    num_edges = sum(edge_type_counts.values())

    sample_object_edges = []
    for object_name, object_spec in config.get("object_nodes", {}).items():
        source_col = object_spec["source_col"]
        if source_col not in trace.columns or prefix_len == 0:
            continue
        node_type = object_spec.get("node_type", object_name)
        edge_type = object_spec.get("edge_type", "interacts")
        value = normalize_value(trace.iloc[0][source_col])
        sample_object_edges.append(f"{node_type}({value})-{edge_type}->event[0]")

    return {
        "case_id": case_id,
        "representation": "object_hetero",
        "prefix_len": prefix_len,
        "case_len": len(trace),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_types": csv_join(list(node_type_counts)),
        "edge_types": csv_join(list(edge_type_counts)),
        "node_type_counts": json_compact(node_type_counts),
        "edge_type_counts": json_compact(edge_type_counts),
        "node_feature_dims": json_compact(node_feature_dims),
        "node_semantics": (
            "There are event nodes for prefix events and object nodes for configured unique "
            f"object values ({csv_join(object_descriptions)})."
        ),
        "edge_semantics": (
            "Follows connects consecutive events. Object edges connect each unique object "
            "value to the events where it appears; reverse edges are added."
        ),
        "feature_columns": csv_join(event_categorical_cols + event_numeric_cols),
        "sample_nodes": csv_join(
            [f"event[0]={normalize_value(trace.iloc[0][activity_col])}"] + [
                f"{node_type}[count]={count}"
                for node_type, count in list(node_type_counts.items())[1:3]
            ]
        ),
        "sample_edges": csv_join(
            ([f"event[0]->event[1]"] if prefix_len > 1 else []) + sample_object_edges[:3]
        ),
        "density": edge_density(num_nodes, num_edges),
        **get_next_event_summary(trace, prefix_len, activity_col, timestamp_col=timestamp_col),
    }


def select_case_ids(df: pd.DataFrame, case_id_col: str, requested_ids: list[str], first_n: int) -> list[str]:
    available_ids = list(df[case_id_col].astype(str).drop_duplicates())
    if requested_ids:
        available_set = set(available_ids)
        missing = [case_id for case_id in requested_ids if case_id not in available_set]
        if missing:
            print(f"Warning: missing case IDs skipped: {csv_join(missing)}")
        return [case_id for case_id in requested_ids if case_id in available_set]
    return available_ids[:first_n]


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)
    if display_df.empty:
        return "_No rows._"

    headers = list(display_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display_df.iterrows():
        values = []
        for col in headers:
            value = normalize_value(row[col])
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def get_case_trace(
    df: pd.DataFrame,
    case_id_col: str,
    case_id: str,
) -> pd.DataFrame:
    return (
        df[df[case_id_col].astype(str) == str(case_id)]
        .reset_index(drop=True)
        .drop(columns=[case_id_col])
    )


def feature_value_summary(
    trace: pd.DataFrame,
    categorical_cols: list[str],
    numerical_cols: list[str],
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        if col not in trace.columns:
            continue
        counts = normalize_series(trace[col]).value_counts(dropna=False)
        rows.append(
            {
                "feature": col,
                "kind": "categorical",
                "non_null": int(trace[col].notna().sum()),
                "missing": int(trace[col].isna().sum()),
                "unique_values": int(counts.shape[0]),
                "top_values": csv_join([f"{idx} ({count})" for idx, count in counts.head(top_n).items()]),
            }
        )
    for col in numerical_cols:
        if col not in trace.columns:
            continue
        numeric = pd.to_numeric(trace[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "kind": "numeric",
                "non_null": int(numeric.notna().sum()),
                "missing": int(numeric.isna().sum()),
                "unique_values": int(numeric.nunique(dropna=True)),
                "top_values": (
                    f"min={numeric.min()}, max={numeric.max()}, "
                    f"mean={round(float(numeric.mean()), 4) if numeric.notna().any() else ''}"
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_block_lines(
    categorical_cols: list[str],
    numerical_cols: list[str],
    category_sizes: dict[str, int],
) -> list[str]:
    lines = []
    for col in categorical_cols:
        lines.append(f"- `{col}`: categorical one-hot block with {category_sizes.get(col, 0)} columns.")
    for col in numerical_cols:
        lines.append(f"- `{col}`: numeric scalar block with 1 column.")
    return lines


def format_dict_as_bullets(title: str, value: dict[str, Any]) -> list[str]:
    lines = [f"**{title}**"]
    for key, item in value.items():
        lines.append(f"- `{key}`: {item}")
    return lines


def render_homo_detail(
    dataset_info: dict[str, Any],
    trace: pd.DataFrame,
    prefix_len: int,
    activity_col: str,
    timestamp_col: str,
    category_sizes: dict[str, int],
) -> str:
    row = summarize_homo(
        dataset_info,
        trace,
        "case",
        prefix_len,
        activity_col,
        timestamp_col,
        category_sizes,
    )
    categorical_cols = [col for col in dataset_info.get("categorical", []) if col in trace.columns]
    numerical_cols = [col for col in dataset_info.get("numerical", []) if col in trace.columns]
    prefix_trace = trace.iloc[:prefix_len]
    lines = [
        "## Homogeneous Graph",
        "",
        "All prefix events become nodes of a single type: `event`.",
        "",
        f"- Prefix length: {prefix_len}",
        f"- Nodes: {row['num_nodes']}",
        f"- Edges after `ToUndirected`: {row['num_edges']}",
        f"- Event feature dimension: {json.loads(row['node_feature_dims'])['event']}",
        f"- Next activity label: `{row['next_activity']}`",
        "",
        "**Event Node Feature Composition**",
        *feature_block_lines(categorical_cols, numerical_cols, category_sizes),
        "",
        "**Node Examples**",
    ]
    for idx in range(min(prefix_len, 5)):
        values = []
        for col in categorical_cols + numerical_cols:
            values.append(f"{col}={normalize_value(prefix_trace.iloc[idx][col])}")
        lines.append(f"- `event[{idx}]`: " + "; ".join(values))
    lines.extend(
        [
            "",
            "**Edge Composition**",
            "- `event -> follows -> event`: event `i` connects to event `i+1`.",
            "- `ToUndirected` adds the reverse direction, so a prefix of length "
            f"{prefix_len} has `{max(prefix_len - 1, 0) * 2}` follows edges.",
            "",
        ]
    )
    return "\n".join(lines)


def render_regular_hetero_detail(
    config: dict[str, Any],
    trace: pd.DataFrame,
    prefix_len: int,
    category_sizes: dict[str, int],
) -> str:
    row = summarize_regular_hetero(config, trace, "case", prefix_len, category_sizes)
    categorical_cols = [col for col in config.get("categorical", []) if col in trace.columns]
    numerical_cols = [col for col in config.get("numerical", []) if col in trace.columns]
    node_cols = list(dict.fromkeys(categorical_cols + numerical_cols))
    activity_col = config["activity_col"]
    node_counts = json.loads(row["node_type_counts"])
    edge_counts = json.loads(row["edge_type_counts"])
    feature_dims = json.loads(row["node_feature_dims"])
    prefix_trace = trace.iloc[:prefix_len]

    lines = [
        "## Regular Heterogeneous Graph",
        "",
        "Each configured feature column becomes its own node type. The graph keeps event "
        "positions aligned by linking each activity node to the other feature nodes at the "
        "same prefix position.",
        "",
        f"- Prefix length: {prefix_len}",
        f"- Total nodes: {row['num_nodes']}",
        f"- Total edges after `ToUndirected`: {row['num_edges']}",
        f"- Activity anchor node type: `{activity_col}`",
        f"- Next activity label: `{row['next_activity']}`",
        "",
        *format_dict_as_bullets("Node Counts", node_counts),
        "",
        "**Node Feature Composition**",
    ]
    for col in node_cols:
        kind = "categorical one-hot" if col in categorical_cols else "numeric scalar"
        lines.append(f"- `{col}` nodes: {kind}, feature dimension `{feature_dims[col]}`.")
    lines.extend(["", "**Edge Type Counts**"])
    for edge_type, count in edge_counts.items():
        lines.append(f"- `{edge_type}`: {count}")
    lines.extend(["", "**Position 0 Node Examples**"])
    if prefix_len > 0:
        for col in node_cols:
            lines.append(f"- `{col}[0]`: value `{normalize_value(prefix_trace.iloc[0][col])}`")
    lines.extend(
        [
            "",
            "**How To Read The Edges**",
            "- `feature -> follows -> feature` connects consecutive prefix positions inside the same feature type.",
            f"- `{activity_col} -> has_<feature> -> feature` connects the activity at a position to the other attributes at that same position.",
            "- Reverse edge types are produced by `ToUndirected`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_object_hetero_detail(
    config: dict[str, Any],
    trace: pd.DataFrame,
    prefix_len: int,
    category_sizes: dict[str, int],
) -> str:
    row = summarize_object_hetero(config, trace, "case", prefix_len, category_sizes)
    prefix_trace = trace.iloc[:prefix_len]
    activity_col = config["activity_col"]
    event_categorical_cols = [
        col for col in config["event_features"].get("categorical", []) if col in trace.columns
    ]
    event_numeric_cols = [
        col for col in config["event_features"].get("numeric", []) if col in trace.columns
    ]
    node_counts = json.loads(row["node_type_counts"])
    edge_counts = json.loads(row["edge_type_counts"])
    feature_dims = json.loads(row["node_feature_dims"])

    lines = [
        "## Object-Centric Heterogeneous Graph",
        "",
        "This representation keeps events as event nodes, but pulls selected attributes out "
        "as object nodes. In the current BPI 2012 config, `org:resource` becomes the "
        "`resource` node type.",
        "",
        f"- Prefix length: {prefix_len}",
        f"- Total nodes: {row['num_nodes']}",
        f"- Total edges after `ToUndirected`: {row['num_edges']}",
        f"- Next activity label: `{row['next_activity']}`",
        "",
        *format_dict_as_bullets("Node Counts", node_counts),
        "",
        "**Event Node Feature Composition**",
        f"- `event` nodes: one node per prefix event, feature dimension `{feature_dims['event']}`.",
        *feature_block_lines(event_categorical_cols, event_numeric_cols, category_sizes),
        "",
    ]

    for object_name, object_spec in config.get("object_nodes", {}).items():
        source_col = object_spec["source_col"]
        if source_col not in trace.columns:
            continue
        node_type = object_spec.get("node_type", object_name)
        edge_type = object_spec.get("edge_type", "interacts")
        values = list(dict.fromkeys(normalize_value(value) for value in prefix_trace[source_col]))
        counts = normalize_series(prefix_trace[source_col]).value_counts()
        lines.extend(
            [
                f"**`{node_type}` Object Node Feature Composition**",
                f"- Source column: `{source_col}`",
                f"- Unique object nodes in this prefix: {len(values)}",
                f"- Feature dimension: `{feature_dims[node_type]}`",
                "- Feature meaning: one-hot encoding of the object value."
                if object_spec.get("encode_source_as_feature", True)
                else "- Feature meaning: constant placeholder feature.",
                f"- Object values: {csv_join(values)}",
                f"- Object occurrence counts: {csv_join([f'{idx} ({count})' for idx, count in counts.items()])}",
                f"- Edge type: `{node_type} -> {edge_type} -> event` connects each object value to every event where it appears.",
                "",
            ]
        )

    lines.extend(["**Event Node Examples**"])
    for idx in range(min(prefix_len, 5)):
        values = []
        for col in event_categorical_cols + event_numeric_cols:
            values.append(f"{col}={normalize_value(prefix_trace.iloc[idx][col])}")
        if "org:resource" in prefix_trace.columns:
            values.append(f"resource_object={normalize_value(prefix_trace.iloc[idx]['org:resource'])}")
        lines.append(f"- `event[{idx}]`: " + "; ".join(values))

    lines.extend(["", "**Edge Type Counts**"])
    for edge_type, count in edge_counts.items():
        lines.append(f"- `{edge_type}`: {count}")
    lines.extend(
        [
            "",
            "**How To Read The Edges**",
            "- `event -> follows -> event` preserves the temporal order of prefix events.",
            "- Object-to-event edges say which object value participates in which event.",
            "- Reverse edge types are produced by `ToUndirected`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> str:
    root = resolve_project_root()
    processed_dir = resolve_dir(
        "THESIS_PROCESSED_DIR",
        root / "data" / "datasets" / "processed",
    )
    dataset_features = load_json(root / "data" / "dataset_features.json")
    hetero_configs = load_json(root / "data" / "hetero_graph_config.json")
    object_configs = load_json(root / "data" / "object_graph_config.json")

    if args.dataset not in dataset_features:
        raise ValueError(f"Dataset '{args.dataset}' is not defined in dataset_features.json.")
    if args.dataset not in hetero_configs:
        raise ValueError(f"Dataset '{args.dataset}' is not defined in hetero_graph_config.json.")

    df = load_processed_split(processed_dir, args.dataset, args.split)
    all_df = load_processed_split(processed_dir, args.dataset, "all")
    hetero_config = hetero_configs[args.dataset]
    object_config = object_configs.get(args.dataset)
    case_id_col = hetero_config["case_id_col"]
    activity_col = hetero_config["activity_col"]
    timestamp_col = hetero_config["timestamp_col"]
    min_prefix_len = int(hetero_config.get("min_prefix_length", 2))

    requested_case_ids = parse_case_ids(args.cases, args.case_file)
    selected_case_ids = select_case_ids(df, case_id_col, requested_case_ids, args.first_n_cases)
    if not selected_case_ids:
        raise ValueError("No case was selected for the report.")
    if len(selected_case_ids) > 1:
        print(f"Warning: report mode uses one case. Using case {selected_case_ids[0]}.")
    case_id = selected_case_ids[0]
    trace = get_case_trace(df, case_id_col, case_id)
    max_prefix_len = len(trace) - 1
    prefix_lengths = parse_prefix_lengths(args.prefix_lengths, min_prefix_len, max_prefix_len)
    if not prefix_lengths:
        raise ValueError(f"Case {case_id} has only {len(trace)} events; no valid prefix was found.")
    prefix_len = prefix_lengths[-1]

    categorical_for_dims = list(dataset_features[args.dataset].get("categorical", []))
    categorical_for_dims.extend(hetero_config.get("categorical", []))
    if object_config:
        categorical_for_dims.extend(object_config["event_features"].get("categorical", []))
        for object_spec in object_config.get("object_nodes", {}).values():
            categorical_for_dims.append(object_spec["source_col"])
    categorical_for_dims = list(dict.fromkeys(categorical_for_dims))
    category_sizes = fit_category_sizes(all_df, categorical_for_dims)

    case_columns = [col for col in trace.columns if col != case_id_col]
    categorical_cols = list(
        dict.fromkeys(
            [col for col in dataset_features[args.dataset].get("categorical", []) if col in trace.columns]
            + [col for col in hetero_config.get("categorical", []) if col in trace.columns]
        )
    )
    numerical_cols = list(
        dict.fromkeys(
            [col for col in dataset_features[args.dataset].get("numerical", []) if col in trace.columns]
            + [col for col in hetero_config.get("numerical", []) if col in trace.columns]
        )
    )

    case_table = trace.copy()
    case_table.insert(0, "event_index", range(len(case_table)))
    summary_df = feature_value_summary(trace, categorical_cols, numerical_cols, args.top_values)
    compact_rows = []
    for representation in ["homo", "hetero", "object_hetero"]:
        if representation == "object_hetero" and object_config is None:
            continue
        compact_args = argparse.Namespace(**vars(args))
        compact_args.cases = str(case_id)
        compact_args.case_file = None
        compact_args.prefix_lengths = str(prefix_len)
        compact_args.representations = [representation]
        compact_rows.extend(build_rows(compact_args))
    compact_df = pd.DataFrame(compact_rows)

    lines = [
        f"# Graph Representation Case Report: {args.dataset} / Case {case_id}",
        "",
        "## Report Settings",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Split: `{args.split}`",
        f"- Case ID column: `{case_id_col}`",
        f"- Activity column: `{activity_col}`",
        f"- Timestamp column: `{timestamp_col}`",
        f"- Case length: {len(trace)} events",
        f"- Inspected prefix length: {prefix_len}",
        f"- Next event index predicted by that prefix: {prefix_len}",
        "",
        "## Full Case",
        "",
        markdown_table(case_table[["event_index"] + case_columns], max_rows=args.max_case_rows),
    ]
    if args.max_case_rows is not None and len(case_table) > args.max_case_rows:
        lines.append(f"\n_Showing the first {args.max_case_rows} events out of {len(case_table)}._")

    lines.extend(
        [
            "",
            "## Feature Value Summary For This Case",
            "",
            markdown_table(summary_df),
            "",
            "## Compact Representation Comparison",
            "",
            markdown_table(
                compact_df[
                    [
                        "representation",
                        "prefix_len",
                        "num_nodes",
                        "num_edges",
                        "node_type_counts",
                        "edge_type_counts",
                        "next_activity",
                    ]
                ]
            ),
            "",
            render_homo_detail(
                dataset_features[args.dataset],
                trace,
                prefix_len,
                activity_col,
                timestamp_col,
                category_sizes,
            ),
            render_regular_hetero_detail(
                hetero_config,
                trace,
                prefix_len,
                category_sizes,
            ),
        ]
    )
    if object_config is not None:
        lines.append(render_object_hetero_detail(object_config, trace, prefix_len, category_sizes))
    else:
        lines.extend(
            [
                "## Object-Centric Heterogeneous Graph",
                "",
                "No object-centric configuration is defined for this dataset.",
            ]
        )
    return "\n".join(lines)


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = resolve_project_root()
    processed_dir = resolve_dir(
        "THESIS_PROCESSED_DIR",
        root / "data" / "datasets" / "processed",
    )
    dataset_features = load_json(root / "data" / "dataset_features.json")
    hetero_configs = load_json(root / "data" / "hetero_graph_config.json")
    object_configs = load_json(root / "data" / "object_graph_config.json")

    if args.dataset not in dataset_features:
        raise ValueError(f"Dataset '{args.dataset}' is not defined in dataset_features.json.")
    if args.dataset not in hetero_configs:
        raise ValueError(f"Dataset '{args.dataset}' is not defined in hetero_graph_config.json.")

    df = load_processed_split(processed_dir, args.dataset, args.split)
    all_df = load_processed_split(processed_dir, args.dataset, "all")
    hetero_config = hetero_configs[args.dataset]
    object_config = object_configs.get(args.dataset)
    case_id_col = hetero_config["case_id_col"]
    activity_col = hetero_config["activity_col"]
    timestamp_col = hetero_config["timestamp_col"]
    min_prefix_len = int(hetero_config.get("min_prefix_length", 2))

    categorical_for_dims = list(dataset_features[args.dataset].get("categorical", []))
    categorical_for_dims.extend(hetero_config.get("categorical", []))
    if object_config:
        categorical_for_dims.extend(object_config["event_features"].get("categorical", []))
        for object_spec in object_config.get("object_nodes", {}).values():
            categorical_for_dims.append(object_spec["source_col"])
    categorical_for_dims = list(dict.fromkeys(categorical_for_dims))
    category_sizes = fit_category_sizes(all_df, categorical_for_dims)

    requested_case_ids = parse_case_ids(args.cases, args.case_file)
    selected_case_ids = select_case_ids(df, case_id_col, requested_case_ids, args.first_n_cases)
    rows = []

    for case_id in selected_case_ids:
        trace = (
            df[df[case_id_col].astype(str) == str(case_id)]
            .reset_index(drop=True)
            .drop(columns=[case_id_col])
        )
        max_prefix_len = len(trace) - 1
        prefix_lengths = parse_prefix_lengths(args.prefix_lengths, min_prefix_len, max_prefix_len)
        if not prefix_lengths:
            print(f"Warning: case {case_id} skipped because it has only {len(trace)} events.")
            continue
        for prefix_len in prefix_lengths:
            if "homo" in args.representations:
                rows.append(
                    summarize_homo(
                        dataset_features[args.dataset],
                        trace,
                        case_id,
                        prefix_len,
                        activity_col,
                        timestamp_col,
                        category_sizes,
                    )
                )
            if "hetero" in args.representations:
                rows.append(
                    summarize_regular_hetero(
                        hetero_config,
                        trace,
                        case_id,
                        prefix_len,
                        category_sizes,
                    )
                )
            if "object_hetero" in args.representations:
                if object_config is None:
                    print(
                        f"Warning: object_hetero skipped for dataset {args.dataset}; "
                        "it is not defined in object_graph_config.json."
                    )
                else:
                    rows.append(
                        summarize_object_hetero(
                            object_config,
                            trace,
                            case_id,
                            prefix_len,
                            category_sizes,
                        )
                    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare homogeneous, regular heterogeneous, and object-centric "
            "heterogeneous event-log graph representations for the same cases."
        )
    )
    parser.add_argument("--dataset", default="bpi_2012", help="Dataset key to inspect.")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "validation", "test"],
        help="Processed split where case IDs are selected.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated CaseID list. If omitted, the first N cases are used.",
    )
    parser.add_argument("--case-file", default=None, help="Text file with one CaseID per line.")
    parser.add_argument("--first-n-cases", type=int, default=3)
    parser.add_argument(
        "--prefix-lengths",
        default="last",
        help="Use 'last', 'all', or a comma-separated list such as '2,5,10'.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["homo", "hetero", "object_hetero"],
        choices=["homo", "hetero", "object_hetero"],
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to export the comparison table as CSV.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Create a detailed English Markdown report for one case.",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Optional path to save the detailed Markdown report.",
    )
    parser.add_argument(
        "--top-values",
        type=int,
        default=8,
        help="Number of most frequent values to show per feature in report mode.",
    )
    parser.add_argument(
        "--max-case-rows",
        type=int,
        default=None,
        help="Limit the number of case events printed in report mode. Default: full case.",
    )
    parser.add_argument("--show-all-columns", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report:
        report = build_report(args)
        if args.report_file:
            output_path = Path(args.report_file).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            print(f"Report exported: {output_path.resolve()}")
        else:
            print(report)
        return

    rows = build_rows(args)
    if not rows:
        print("No comparison rows were created.")
        return

    df = pd.DataFrame(rows)
    sort_cols = ["case_id", "prefix_len", "representation"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if args.csv:
        output_path = Path(args.csv).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"CSV exported: {output_path.resolve()}")

    if args.show_all_columns:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 240)
        print(df.to_string(index=False))
    else:
        columns = [
            "case_id",
            "representation",
            "prefix_len",
            "case_len",
            "num_nodes",
            "num_edges",
            "node_type_counts",
            "edge_type_counts",
            "next_activity",
        ]
        print(df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
