---
tags: [quickstart, tabular, federated-learning]
dataset: [PaySim, Fed-Fraud]
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

---

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @yan-gao/fed-fin-fraud
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

---

## Run the App

You can run this Flower App in both **simulation** and **deployment** mode.

---

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

---

### Model

The model is a **fully connected neural network (MLP)**:

* Input: engineered tabular features
* Two hidden layers with:

  * LayerNorm (optional)
  * ReLU activation
  * Dropout
* Output: single logit for binary classification

Defined in: 

---

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

---

### Training

Each client:

* Receives global model weights
* Trains locally using:

  * `BCEWithLogitsLoss`
  * Optional class weighting
  * Gradient clipping
* Uses **cosine annealing learning rate schedule**

Training logic: 

---

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

Evaluation logic: 

---

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

---

## Notes

* Designed for **highly imbalanced fraud detection**
* Uses **PR-AUC** as a key evaluation metric
* Supports both research (simulation) and real-world deployment
* Automatically uses GPU if available

---
