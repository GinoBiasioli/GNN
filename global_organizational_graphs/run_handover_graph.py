import json
import math
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from global_organizational_graphs.algorithm import apply
from global_organizational_graphs.variants.dataframe import Parameters


DATASET_NAME = "bpi_2012"
CASE_ID_COLUMN = "CaseID"
TIMESTAMP_COLUMN = "time:timestamp"
RESOURCE_COLUMN = "org:resource"
ACTIVITY_COLUMN = "concept:name"

#DATASET_NAME = "bpi_2013"
#CASE_ID_COLUMN = "CaseID"
#TIMESTAMP_COLUMN = "time:timestamp"
#RESOURCE_COLUMN = "org:resource"
#ACTIVITY_COLUMN = "Activity"

#DATASET_NAME = "sp2020"
#CASE_ID_COLUMN = "CaseID"
#TIMESTAMP_COLUMN = "time:timestamp"
#RESOURCE_COLUMN = "org:resource"
#ACTIVITY_COLUMN = "Activity"

#DATASET_NAME = "BPI20_RequestForPayment"
#CASE_ID_COLUMN = "CaseID"
#TIMESTAMP_COLUMN = "time:timestamp"
#RESOURCE_COLUMN = "org:role"
#ACTIVITY_COLUMN = "Activity"

# GraphML edge attributes. The enriched CSV/JSON outputs always keep every
# metric, while these flags control how much information is written to the graph.
GRAPH_INCLUDE_SOURCE_TRANSITION_PROBABILITY = True
GRAPH_INCLUDE_MEDIAN_TIME_BETWEEN_EVENTS_NORMALIZED = True
GRAPH_INCLUDE_ACTIVITY_ENTROPY_NORMALIZED = True

GRAPH_INCLUDE_ACTIVITY_BREAKDOWN_OUT = False
GRAPH_INCLUDE_ACTIVITY_BREAKDOWN_IN = False
GRAPH_INCLUDE_DOMINANT_ACTIVITY_OUT = False
GRAPH_INCLUDE_DOMINANT_ACTIVITY_IN = False
GRAPH_INCLUDE_AVG_TIME_BETWEEN_EVENTS = False
GRAPH_INCLUDE_MEDIAN_TIME_BETWEEN_EVENTS = False
GRAPH_INCLUDE_ACTIVITY_ENTROPY = False


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_input_path(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "datasets"
        / "processed"
        / DATASET_NAME
        / f"{DATASET_NAME}_processed_train.csv"
    )


def get_output_dir() -> Path:
    output_dir = Path(__file__).resolve().parent / "outputs" / DATASET_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def collapse_activity_counts(network_edges: dict) -> list[dict]:
    """
    Collapse the PM4Py edge-type breakdown into one total handover frequency per resource pair.
    """
    rows = []
    for (source_resource, target_resource), activity_counts in network_edges.items():
        rows.append(
            {
                "source_resource": source_resource,
                "target_resource": target_resource,
                "handover_frequency": int(sum(activity_counts.values())),
                "activity_breakdown": json.dumps(activity_counts, ensure_ascii=False),
            }
        )
    return rows


def get_dominant_activity(activity_counts: dict) -> str | None:
    if not activity_counts:
        return None
    return max(activity_counts.items(), key=lambda item: item[1])[0]


def get_activity_entropy(activity_counts: dict) -> float:
    total = sum(activity_counts.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in activity_counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def get_activity_entropy_normalized(activity_counts: dict) -> float:
    entropy = get_activity_entropy(activity_counts)
    number_of_activities = len(activity_counts)
    if number_of_activities <= 1:
        return 0.0
    return entropy / math.log2(number_of_activities)


def add_relative_edge_metrics(rows: list[dict]) -> list[dict]:
    outgoing_weight_by_source = {}
    for row in rows:
        source_resource = row["source_resource"]
        outgoing_weight_by_source[source_resource] = (
            outgoing_weight_by_source.get(source_resource, 0) + row["weight"]
        )

    median_log_values = [
        math.log1p(max(row["median_time_between_events"], 0.0)) for row in rows
    ]
    min_median_log = min(median_log_values, default=0.0)
    max_median_log = max(median_log_values, default=0.0)
    median_log_range = max_median_log - min_median_log

    for row, median_log_value in zip(rows, median_log_values):
        total_outgoing = outgoing_weight_by_source[row["source_resource"]]
        row["source_transition_probability"] = row["weight"] / total_outgoing
        row["median_time_between_events_log"] = median_log_value

        if median_log_range > 0:
            row["median_time_between_events_normalized"] = (
                median_log_value - min_median_log
            ) / median_log_range
        else:
            row["median_time_between_events_normalized"] = 0.0

    return rows


def build_enriched_handover_rows(dataframe: pd.DataFrame) -> list[dict]:
    """
    Build resource-to-resource handover rows with source/target activity context
    and waiting-time metrics for each directly-follows event pair within a case.
    """
    required_columns = [
        CASE_ID_COLUMN,
        TIMESTAMP_COLUMN,
        RESOURCE_COLUMN,
        ACTIVITY_COLUMN,
    ]
    handover_df = dataframe[required_columns].copy()
    handover_df = handover_df.dropna(subset=required_columns)
    handover_df = handover_df.sort_values(
        by=[CASE_ID_COLUMN, TIMESTAMP_COLUMN],
        kind="mergesort",
    )

    grouped = handover_df.groupby(CASE_ID_COLUMN, sort=False)
    handover_df["target_resource"] = grouped[RESOURCE_COLUMN].shift(-1)
    handover_df["target_activity"] = grouped[ACTIVITY_COLUMN].shift(-1)
    handover_df["target_timestamp"] = grouped[TIMESTAMP_COLUMN].shift(-1)

    handover_df = handover_df.dropna(
        subset=["target_resource", "target_activity", "target_timestamp"]
    )
    handover_df["time_between_events"] = (
        handover_df["target_timestamp"] - handover_df[TIMESTAMP_COLUMN]
    ).dt.total_seconds()

    rows = []
    for (source_resource, target_resource), edge_df in handover_df.groupby(
        [RESOURCE_COLUMN, "target_resource"],
        sort=False,
    ):
        activity_breakdown_out = {
            str(activity): int(count)
            for activity, count in edge_df[ACTIVITY_COLUMN].value_counts().items()
        }
        activity_breakdown_in = {
            str(activity): int(count)
            for activity, count in edge_df["target_activity"].value_counts().items()
        }
        time_between_events = edge_df["time_between_events"]

        rows.append(
            {
                "source_resource": source_resource,
                "target_resource": target_resource,
                "weight": int(len(edge_df)),
                "activity_breakdown_out": json.dumps(
                    activity_breakdown_out,
                    ensure_ascii=False,
                ),
                "activity_breakdown_in": json.dumps(
                    activity_breakdown_in,
                    ensure_ascii=False,
                ),
                "dominant_activity_out": get_dominant_activity(
                    activity_breakdown_out
                ),
                "dominant_activity_in": get_dominant_activity(activity_breakdown_in),
                "avg_time_between_events": float(time_between_events.mean()),
                "median_time_between_events": float(time_between_events.median()),
                "activity_entropy": float(get_activity_entropy(activity_breakdown_out)),
                "activity_entropy_normalized": float(
                    get_activity_entropy_normalized(activity_breakdown_out)
                ),
            }
        )
    return add_relative_edge_metrics(rows)


def get_graph_edge_attributes(row: dict) -> dict:
    attributes = {"weight": row["weight"]}

    if GRAPH_INCLUDE_SOURCE_TRANSITION_PROBABILITY:
        attributes["source_transition_probability"] = row[
            "source_transition_probability"
        ]
    if GRAPH_INCLUDE_MEDIAN_TIME_BETWEEN_EVENTS_NORMALIZED:
        attributes["median_time_between_events_normalized"] = row[
            "median_time_between_events_normalized"
        ]
    if GRAPH_INCLUDE_ACTIVITY_ENTROPY_NORMALIZED:
        attributes["activity_entropy_normalized"] = row[
            "activity_entropy_normalized"
        ]
    if GRAPH_INCLUDE_ACTIVITY_BREAKDOWN_OUT:
        attributes["activity_breakdown_out"] = row["activity_breakdown_out"]
    if GRAPH_INCLUDE_ACTIVITY_BREAKDOWN_IN:
        attributes["activity_breakdown_in"] = row["activity_breakdown_in"]
    if GRAPH_INCLUDE_DOMINANT_ACTIVITY_OUT:
        attributes["dominant_activity_out"] = row["dominant_activity_out"]
    if GRAPH_INCLUDE_DOMINANT_ACTIVITY_IN:
        attributes["dominant_activity_in"] = row["dominant_activity_in"]
    if GRAPH_INCLUDE_AVG_TIME_BETWEEN_EVENTS:
        attributes["avg_time_between_events"] = row["avg_time_between_events"]
    if GRAPH_INCLUDE_MEDIAN_TIME_BETWEEN_EVENTS:
        attributes["median_time_between_events"] = row["median_time_between_events"]
    if GRAPH_INCLUDE_ACTIVITY_ENTROPY:
        attributes["activity_entropy"] = row["activity_entropy"]

    return attributes


def main() -> None:
    project_root = get_project_root()
    input_path = get_input_path(project_root)
    output_dir = get_output_dir()

    # The global organizational graph is built from the training set only.
    train_df = pd.read_csv(input_path)

    # The processed datasets store timestamps as numeric day offsets.
    # PM4Py's network analysis internally computes timestamp differences
    # with pandas' datetime accessors, so we cast the offsets to datetimes
    # while preserving their order and pairwise distances.
    train_df[TIMESTAMP_COLUMN] = pd.to_datetime(
        train_df[TIMESTAMP_COLUMN],
        unit="D",
        origin="unix",
    )

    # Handover-of-work setup:
    # - OUT and IN columns use the case identifier, so events are linked within the same case.
    # - SOURCE and TARGET node columns both use org:resource, so graph nodes are resources.
    # - EDGE column stores the activity of the source event; PM4Py keeps this breakdown in the output.
    # - SORTING column uses the timestamp, so the handover is based on event succession over time.
    parameters = {
        Parameters.OUT_COLUMN: CASE_ID_COLUMN,
        Parameters.IN_COLUMN: CASE_ID_COLUMN,
        Parameters.NODE_COLUMN_SOURCE: RESOURCE_COLUMN,
        Parameters.NODE_COLUMN_TARGET: RESOURCE_COLUMN,
        Parameters.EDGE_COLUMN: ACTIVITY_COLUMN,
        Parameters.SORTING_COLUMN: TIMESTAMP_COLUMN,
        Parameters.TIMESTAMP_KEY: TIMESTAMP_COLUMN,
        Parameters.INCLUDE_PERFORMANCE: False,
    }

    network_edges = apply(train_df, parameters=parameters)

    raw_output_path = output_dir / f"{DATASET_NAME}_handover_network_raw.json"
    raw_serializable = {
        f"{source}|||{target}": edge_values
        for (source, target), edge_values in network_edges.items()
    }
    raw_output_path.write_text(
        json.dumps(raw_serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    collapsed_rows = collapse_activity_counts(network_edges)
    collapsed_df = pd.DataFrame(collapsed_rows).sort_values(
        by="handover_frequency",
        ascending=False,
    )
    collapsed_output_path = output_dir / f"{DATASET_NAME}_handover_edges.csv"
    collapsed_df.to_csv(collapsed_output_path, index=False)

    enriched_rows = build_enriched_handover_rows(train_df)
    enriched_df = pd.DataFrame(enriched_rows).sort_values(
        by="weight",
        ascending=False,
    )
    enriched_output_path = output_dir / f"{DATASET_NAME}_handover_edges_enriched.csv"
    enriched_df.to_csv(enriched_output_path, index=False)

    enriched_json_output_path = (
        output_dir / f"{DATASET_NAME}_handover_edges_enriched.json"
    )
    enriched_json_output_path.write_text(
        json.dumps(enriched_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    graph = nx.DiGraph()
    for row in enriched_rows:
        graph.add_edge(
            row["source_resource"],
            row["target_resource"],
            **get_graph_edge_attributes(row),
        )
    graphml_output_path = output_dir / f"{DATASET_NAME}_handover_graph.graphml"
    nx.write_graphml(graph, graphml_output_path)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Input: {input_path}")
    print(f"Raw PM4Py-style output: {raw_output_path}")
    print(f"Collapsed handover frequencies: {collapsed_output_path}")
    print(f"Enriched handover edges CSV: {enriched_output_path}")
    print(f"Enriched handover edges JSON: {enriched_json_output_path}")
    print(f"Directed graph file: {graphml_output_path}")
    print(f"Number of resource-to-resource edges: {len(enriched_rows)}")


if __name__ == "__main__":
    main()
