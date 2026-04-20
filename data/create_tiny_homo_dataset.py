import os
from pathlib import Path

import torch


def resolve_project_root():
    env_root = os.environ.get("THESIS_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "data" / "dataset_features.json").exists():
            return candidate
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
            return candidate

    raise FileNotFoundError(
        "Could not locate the thesis project root. "
        "Expected to find data/dataset_features.json."
    )


ROOT_PATH = resolve_project_root()
GRAPHS_ENV = os.environ.get("THESIS_GRAPHS_DIR")
if GRAPHS_ENV:
    DATA_DIR_GRAPHS = Path(os.path.abspath(os.path.expanduser(GRAPHS_ENV)))
else:
    DATA_DIR_GRAPHS = ROOT_PATH / "data" / "datasets" / "hom_graphs"

SOURCE_DATASET = "sp2020"
TARGET_DATASET = "tiny_sp2020"
SPLIT_LIMITS = {
    "train_set_homo.pt": 64,
    "validation_set_homo.pt": 32,
    "test_set_homo.pt": 32,
}


def build_tiny_split(file_name, limit):
    source_path = DATA_DIR_GRAPHS / SOURCE_DATASET / file_name
    target_path = DATA_DIR_GRAPHS / TARGET_DATASET / file_name

    graphs = torch.load(source_path, weights_only=False)
    tiny_graphs = graphs[:limit]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tiny_graphs, target_path)

    print(
        f"{file_name}: saved {len(tiny_graphs)} of {len(graphs)} graphs "
        f"to {target_path}"
    )


def main():
    print(f"Project root: {ROOT_PATH}")
    print(f"Source dataset: {SOURCE_DATASET}")
    print(f"Target dataset: {TARGET_DATASET}")

    for file_name, limit in SPLIT_LIMITS.items():
        build_tiny_split(file_name, limit)

    print("\nTiny dataset created successfully.")


if __name__ == "__main__":
    main()
