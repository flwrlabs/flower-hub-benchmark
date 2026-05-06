---
tags: [benchmark, medical-imaging, federated-learning]
dataset: [fed-brats]
framework: [torch, monai, flwr]
---

# Federated Brain Tumor Segmentation with Flower and MONAI

This example demonstrates how to perform **federated learning for 3D brain tumor segmentation** using **Flower**, **PyTorch**, and **MONAI**. It uses the **fed-brats dataset** hosted on Hugging Face and supports both simulation and deployment workflows.

The project includes:

* A **3D U-Net model** for volumetric segmentation
* Data loading pipelines using **MONAI transforms**
* Federated training with **Flower**
* Support for both **IID and natural (site-based) partitioning**

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/fed-med-seg
```

Then, install dependencies:

```bash
cd fed-med-seg && pip install -e .
```

Project structure:

```shell
fed-med-seg
├── fed_med_seg
│   ├── __init__.py
│   ├── client_app.py   # Client-side training logic
│   ├── server_app.py   # Server-side orchestration and evaluation
│   └── task.py         # Model, data loading, training, evaluation
├── pyproject.toml      # Dependencies and configuration
└── README.md
```

## Run the App

This Flower App supports both **simulation mode** and **deployment mode** without code changes.

### Run with the Simulation Engine

In simulation mode:

* The **fed-brats dataset** is automatically downloaded from Hugging Face
* Data is partitioned across clients using:

  * `iid` (random split), or
  * `natural` (by hospital/site)

Run with default configuration:

```bash
flwr run .
```

Override configuration (example):

```bash
flwr run . --run-config "num-server-rounds=5 batch-size=2"
```

Key configurable parameters (from `pyproject.toml`):

* `num-server-rounds`: number of FL rounds
* `local-epochs`: local training epochs per client
* `batch-size`: training batch size
* `roi-x/y/z`: 3D crop size for training
* `partitioner`: `iid` or `natural`
* `learning-rate-max/min`: cosine annealing schedule

### Model

The model is a **3D U-Net** implemented using MONAI:

* Input channels: 4 MRI modalities (`t1n`, `t1c`, `t2w`, `t2f`)
* Output channels: segmentation classes (default: 4)
* Architecture: encoder-decoder with residual units

### Data Pipeline

Data is loaded from the Hugging Face dataset:

* Dataset: `flwrlabs/fed-brats`
* Automatically downloaded and cached locally
* Converted into MONAI-compatible format

Preprocessing includes:

* Resampling to 1mm spacing
* Intensity normalization
* Label remapping
* Random cropping and augmentation (training only)

Two modes:

* **Simulation mode** → uses `FederatedDataset`
* **Deployment mode** → loads pre-partitioned data from disk

### Training

Each client:

* Receives the global model
* Trains locally using:

  * Dice + Cross Entropy loss
  * Adam optimizer
* Applies **cosine annealing learning rate**

### Evaluation

Server-side evaluation:

* Uses centralized **test split**
* Applies sliding window inference
* Reports:

  * Loss
  * Mean Dice score

### Run with the Deployment Engine

For deployment, you must provide local dataset partitions.

#### Step 1: Prepare data

Download and partition the dataset manually (or via Flower Datasets), then store partitions locally.

#### Step 2: Start SuperNodes

Each node must point to its local data:

```shell
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/local_partition"
```

#### Step 3: Run the federation

```shell
flwr run . <SUPERLINK-CONNECTION> --stream
```

## Benchmarking and System Metrics

This app writes a benchmark summary next to the standard Flower result pickle:

```text
result_<run-name>_communication.json
```

The summary includes per-round and total communication volume:

* `total_comm_bytes`
* `comm_bytes_total` per training round

Enable system metric tracking with:

```bash
flwr run . <SUPERLINK-CONNECTION> --stream --run-config "benchmark-system-metrics=true"
```

When enabled, the benchmark summary also includes:

* `client_train_time_sec`
* `server_aggregation_time_sec`
* `round_wall_clock_sec`
* `client_peak_cpu_memory_mb`
* `client_peak_gpu_memory_mb`

Server-side centralized evaluation can be disabled for benchmark-only runs:

```bash
flwr run . <SUPERLINK-CONNECTION> --stream --run-config "benchmark-run-server-eval=false"
```

### Deployment Fingerprints

Use the following fed-brats partition fingerprints when checking that each deployment client has the expected local data partition:

| Client | Partition ID | Examples | Dataset fingerprint |
|---:|---:|---:|---|
| 0 | 0 | 379 | `dfad4dd6243e677e2cd6c4d4e20ffa00e03ca61c5f72cbef2d68b43a12b2ef19` |
| 1 | 1 | 25 | `5c1b805391929fb7a6fb787f540fe32f0383ebb6e88c2a8cfecbff85e18e1c37` |
| 2 | 2 | 256 | `d2a719dbbc5c7df5c60e812ab0d3b2737ebca94c1be12a3a9375032c074c4df2` |
| 3 | 3 | 160 | `95d047fbf66c27ad8bdb47655df92314cef493723b2b8bebdc557b093182a0af` |
| 4 | 4 | 477 | `177b73c74d44d6ad92fb511ae1d56a5aa7c6b68e93ea0838b6a3f634feab13c3` |

## Notes

* GPU is automatically used if available
* Large 3D volumes are handled via **sliding window inference**
* Data loading uses MONAI `CacheDataset` for efficiency
* Supports both research (simulation) and real-world deployment
