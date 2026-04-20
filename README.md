# GNN Thesis Code

This repository contains the homogeneous GNN training pipeline for next-event prediction on process mining event logs.

The current training script predicts the next:

- `Activity`
- `org:resource`
- `time:timestamp`

The model uses homogeneous graphs stored as PyTorch Geometric `.pt` files.

## Repository layout

```text
.
├── main_homo.py
├── EDA.py
├── requirements.txt
└── data/
    ├── dataset_features.json
    ├── dataprocessing_hom.py
    ├── create_tiny_homo_dataset.py
    └── utils.py
```

## Data expectations

This repository does not include the large graph datasets.

Before running `main_homo.py`, download the graph folders so that the local structure becomes:

```text
data/
  datasets/
    hom_graphs/
      bpi_2012/
      bpi_2013/
      BPI20_RequestForPayment/
      sp2020/
      tiny_sp2020/
```

If you are using Vertex AI, the intended workflow is:

1. Clone this repository in JupyterLab.
2. Download `hom_graphs/` from Google Cloud Storage.
3. Place it under `./data/datasets/hom_graphs`.
4. Run `main_homo.py`.

## Minimal files needed to train from existing graphs

If graphs are already created, you only need:

- `main_homo.py`
- `data/dataset_features.json`
- `data/datasets/hom_graphs/...`

You do not need processed CSV files unless you want to regenerate graphs.

## Regenerating graphs

To regenerate homogeneous graphs from processed CSVs:

```powershell
py ".\data\dataprocessing_hom.py"
```

To create the tiny test dataset from `sp2020`:

```powershell
py ".\data\create_tiny_homo_dataset.py"
```

## Running training

Select the dataset inside `main_homo.py`, for example:

```python
dataset = "tiny_sp2020"
```

Then run:

```powershell
py ".\main_homo.py"
```

## Path handling

The scripts resolve paths relative to the repository by default, so they work both locally and in Vertex AI after cloning.

Optional environment variables are also supported:

- `THESIS_PROJECT_ROOT`
- `THESIS_PROCESSED_DIR`
- `THESIS_GRAPHS_DIR`
- `THESIS_RESULTS_DIR`

## Main dependencies

- `torch`
- `torch-geometric`
- `ax-platform`
- `torcheval`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `networkx`
- `scipy`
- `tqdm`
