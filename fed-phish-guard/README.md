---
tags: [quickstart, cybersecurity, federated-learning]
dataset: [Phishing URLs]
framework: [torch, flwr]
---

# Federated Phishing URL Detection with Flower and PyTorch

This project implements **federated learning for phishing URL detection** using **Flower** and **PyTorch**.

It combines:

* A **CNN-based text model** for URL classification
* **Byte-level encoding** of URLs
* Federated training across distributed clients
* Support for both **simulation** and **deployment** modes

The system is designed to handle **imbalanced data** and realistic **client-level data partitioning**.

---

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @yan-gao/fed-phish-guard
```

Then, install dependencies:

```bash
cd fed-phish-guard && pip install -e .
```

Project structure:

```shell
fed-phish-guard
├── phishguard
│   ├── __init__.py
│   ├── client_app.py   # Client-side training logic
│   ├── server_app.py   # Server-side orchestration and evaluation
│   ├── model.py        # CNN model for URL classification
│   ├── train.py        # Training and evaluation loops
│   └── data.py         # Data loading and preprocessing
├── pyproject.toml
```

---

## Run the App

This Flower App supports both **simulation** and **deployment** workflows.

---

### Run with the Simulation Engine

In simulation mode:

* Dataset is automatically loaded from Hugging Face (`yangao381/fed-phishing-urls`)
* Data is partitioned **naturally by client_id**

Run with default settings:

```bash
flwr run .
```

Override configuration:

```bash
flwr run . --run-config "num-server-rounds=10 batch-size=64"
```

Key configuration options (from `pyproject.toml`):

* `num-server-rounds`: number of FL rounds
* `local-epochs`: local training epochs
* `batch-size`: batch size
* `embed-dim`: embedding size
* `num-filters`: CNN filters
* `dropout`: dropout rate
* `learning-rate-max/min`: cosine annealing schedule
* `fraction-train`: fraction of clients participating per round

---

## Model

The model is a **CNN-based architecture for URL classification**:

* Input: byte-level encoded URLs
* Embedding layer for byte tokens
* Parallel multi-scale convolutions (kernel sizes 3, 5, 7)
* Stacked convolutional blocks with pooling
* Global max pooling
* Fully connected classifier

Defined in: 

---

## Data Pipeline

Dataset:

* Hugging Face: `flwrlabs/fed-phishing-urls`

Processing steps:

1. Convert URLs → byte sequences
2. Map bytes to indices (vocabulary size = 258)
3. Pad/truncate to fixed length (default: 256)
4. Create PyTorch datasets

Supports:

* **Simulation mode** → federated partitions via `FederatedDataset`
* **Deployment mode** → load local datasets from disk

Class imbalance handling:

* Weighted sampling per client
* Positive class weighting for loss

Implemented in: 

---

## Training

Each client:

* Receives global model weights
* Trains locally using:

  * `BCEWithLogitsLoss`
  * Gradient clipping
  * AdamW optimizer
* Applies **cosine annealing learning rate**

Training loop defined in: 

After each round, clients return:

* Updated model weights
* Aggregated training metrics (loss, accuracy, F1)

---

## Evaluation

Server-side evaluation:

* Uses centralized **test split**
* Reports:

  * Loss
  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC

Evaluation logic: 

---

## Run with the Deployment Engine

To run in deployment mode:

### Step 1: Prepare local datasets

Prepare datasets in Hugging Face format (`Dataset` or `DatasetDict` with `train` split).

### Step 2: Start SuperNodes

```shell
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/local_dataset"
```

### Step 3: Run federation

```shell
flwr run . <SUPERLINK-CONNECTION> --stream
```

---

## Notes

* Designed for **cybersecurity / phishing detection tasks**
* Uses **byte-level encoding** to handle arbitrary URLs
* Handles **class imbalance** via sampling and loss weighting
* Automatically uses GPU if available
* Efficient for variable-length text inputs

---
