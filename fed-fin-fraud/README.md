---
tags: [benchmark, tabular, federated-learning]
dataset: [fed-fraud-paysim-banks]
framework: [torch, flwr]
---

# Federated Financial Fraud Detection with PyTorch and Flower

This example demonstrates **federated learning for financial fraud detection** using **Flower** and **PyTorch** on a **tabular transaction dataset**.

It uses a federated version of the PaySim dataset (`flwrlabs/fed-fraud-paysim-banks`) and supports both:

* **IID partitioning**
* **Natural partitioning by bank (BankID)**

The project includes:

* A **multi-layer perceptron (MLP)** model with LayerNorm and dropout
* Feature preprocessing (normalization + feature engineering + one-hot encoding)
* Handling of **class imbalance** via weighted sampling and loss weighting
* Evaluation across **multiple classification thresholds**

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/fed-fin-fraud
```

Then, install dependencies:

```bash
cd fed-fin-fraud && pip install -e .
```

Project structure:

```shell
fed-fin-fraud
├── fed_fraud
│   ├── __init__.py
│   ├── client_app.py   # Client-side training logic
│   ├── server_app.py   # Server-side orchestration and evaluation
│   └── task.py         # Model, preprocessing, training, evaluation
├── pyproject.toml      # Dependencies and configuration
└── README.md
```

## Run the App

You can run this Flower App in both **simulation** and **deployment** mode.

### Run with the Simulation Engine

In simulation mode:

* Dataset is automatically loaded from Hugging Face
* Training data is partitioned across clients:

  * `iid` → random split
  * `natural` → grouped by `BankID`

Run with default configuration:

```bash
flwr run .
```

Override configuration (example):

```bash
flwr run . --run-config "num-server-rounds=5 batch-size=512"
```

Key configuration options (from `pyproject.toml`):

* `num-server-rounds`: number of FL rounds
* `local-epochs`: local training epochs
* `batch-size`: training batch size
* `hidden-dim-1`, `hidden-dim-2`: model size
* `dropout`: dropout rate
* `use-class-weights`: handle class imbalance
* `partitioner`: `iid` or `natural`
* `learning-rate-max/min`: cosine annealing schedule

### Model

The model is a **fully connected neural network (MLP)**:

* Input: engineered tabular features
* Two hidden layers with:

  * LayerNorm (optional)
  * ReLU activation
  * Dropout
* Output: single logit for binary classification

### Data Pipeline

Dataset:

* `flwrlabs/fed-fraud-paysim-banks`

Processing steps:

1. **Standardization** of numeric features
2. **Feature engineering**:

   * balance deltas
   * transaction inconsistencies
3. **One-hot encoding** of transaction type
4. Construction of final feature vector

Class imbalance handling:

* Weighted sampling (`WeightedRandomSampler`)
* Optional `pos_weight` in loss function

Supports:

* Simulation mode via `FederatedDataset`
* Deployment mode via `load_from_disk`

### Training

Each client:

* Receives global model weights
* Trains locally using:

  * `BCEWithLogitsLoss`
  * Optional class weighting
  * Gradient clipping
* Uses **cosine annealing learning rate schedule**

### Evaluation

Server-side evaluation:

* Uses centralized **test split**
* Computes metrics at multiple thresholds:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * PR-AUC (average precision)

Thresholds evaluated:

```
0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99
```

### Run with the Deployment Engine

To run in deployment mode, prepare local dataset partitions.

#### Step 1: Prepare data

Partition and store the dataset locally (e.g., using Flower Datasets or custom pipeline).

#### Step 2: Start SuperNodes

```shell
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/local_partition"
```

#### Step 3: Run federation

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

### Dataset Fingerprint Verification

FedFinFraud supports a preflight dataset fingerprint check before training. Enable it with:

```bash
flwr run . <SUPERLINK-CONNECTION> --stream --run-config "benchmark-verify-dataset=true"
```

The server asks each connected client for its partition metadata, then verifies:

* expected client count
* partition IDs
* dataset version
* number of examples
* dataset fingerprint

The verification result is written into `result_<run-name>_communication.json` under `verification`. If any partition does not match the benchmark manifest, the run fails before training.

Expected deployment fingerprints for `flwrlabs/fed-fraud-paysim-banks` with natural partitioning:

| Client | Partition ID | Examples | Dataset fingerprint |
|---:|---:|---:|---|
| 0 | 0 | 1374326 | `c35c860d5f75655c0b53312ba8a5b555eeb97b20ac3610b8aabec724e69ef1f7` |
| 1 | 1 | 1202535 | `b3cc17127b78219d7a8d18c60bb88fc59f041012acef3ef5efa37a64ba42cd3f` |
| 2 | 2 | 1145272 | `d311265f241964508546973919ee804d1f19615d464cf5819f40e6a1daf06d06` |
| 3 | 3 | 1030745 | `3c22c4e9d5ffebed3076b86bbfa330f9425431237a4c74a8ce06511346c821bc` |
| 4 | 4 | 973480 | `cf41d0946d13478a69785086a252ac6a4d0e1a1ba77943eb587b77abdba647f9` |

## Notes

* Designed for **highly imbalanced fraud detection**
* Uses **PR-AUC** as a key evaluation metric
* Supports both research (simulation) and real-world deployment
* Automatically uses GPU if available
