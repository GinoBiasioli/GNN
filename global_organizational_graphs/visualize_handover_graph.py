import json
from pathlib import Path

import pm4py




#DATASET_NAME = "bpi_2012"

# DATASET_NAME = "bpi_2013"
#DATASET_NAME = "sp2020"
DATASET_NAME = "BPI20_RequestForPayment"


# Raise this value if the rendered graph is too dense.
EDGE_THRESHOLD = 20
OUTPUT_FORMAT = "svg"


def get_raw_output_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "outputs"
        / DATASET_NAME
        / f"{DATASET_NAME}_handover_network_raw.json"
    )


def load_network_edges(raw_output_path: Path) -> dict:
    """
    Reload the PM4Py-style network-analysis output saved by run_handover_graph.py.
    JSON stores tuple keys as strings, so they are reconstructed here.
    """
    raw_data = json.loads(raw_output_path.read_text(encoding="utf-8"))
    network_edges = {}
    for serialized_key, edge_values in raw_data.items():
        source_resource, target_resource = serialized_key.split("|||", maxsplit=1)
        network_edges[(source_resource, target_resource)] = edge_values
    return network_edges


def main() -> None:
    raw_output_path = get_raw_output_path()
    network_edges = load_network_edges(raw_output_path)

    # Visualize the already generated handover graph without rebuilding it.
    pm4py.view_network_analysis(
        network_edges,
        variant="frequency",
        format=OUTPUT_FORMAT,
        edge_threshold=EDGE_THRESHOLD,
    )


if __name__ == "__main__":
    main()
