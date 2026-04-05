---
tags: [quickstart, medical-imaging, federated-learning]
dataset: [BraTS, Fed-BraTS]
framework: [torch, monai, flwr]
---

# Federated Brain Tumor Segmentation with Flower and MONAI

This example demonstrates how to perform **federated learning for 3D brain tumor segmentation** using **Flower**, **PyTorch**, and **MONAI**. It uses the **Fed-BraTS dataset** hosted on Hugging Face and supports both simulation and deployment workflows.

The project includes:

* A **3D U-Net model** for volumetric segmentation
* Data loading pipelines using **MONAI transforms**
* Federated training with **Flower**
* Support for both **IID and natural (site-based) partitioning**

---

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @yan-gao/fed-med-seg
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

---

## Run the App

This Flower App supports both **simulation mode** and **deployment mode** without code changes.

---

### Run with the Simulation Engine

In simulation mode:

* The **Fed-BraTS dataset** is automatically downloaded from Hugging Face
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

---

### Model

The model is a **3D U-Net** implemented using MONAI:

* Input channels: 4 MRI modalities (`t1n`, `t1c`, `t2w`, `t2f`)
* Output channels: segmentation classes (default: 4)
* Architecture: encoder-decoder with residual units

See implementation in: 

---

### Data Pipeline

Data is loaded from the Hugging Face dataset:

* Dataset: `flwrlabs/Fed-BraTS`
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

---

### Training

Each client:

* Receives the global model
* Trains locally using:

  * Dice + Cross Entropy loss
  * Adam optimizer
* Applies **cosine annealing learning rate**

Training logic is defined in: 

---

### Evaluation

Server-side evaluation:

* Uses centralized **test split**
* Applies sliding window inference
* Reports:

  * Loss
  * Mean Dice score

Implemented in: 

---

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

---

## Notes

* GPU is automatically used if available
* Large 3D volumes are handled via **sliding window inference**
* Data loading uses MONAI `CacheDataset` for efficiency
* Supports both research (simulation) and real-world deployment

---
