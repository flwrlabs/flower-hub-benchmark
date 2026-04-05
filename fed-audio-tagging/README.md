---
tags: [quickstart, audio, federated-learning]
dataset: [UrbanSound8K]
framework: [torch, torchaudio, flwr]
---

# Federated Audio Tagging with Flower and PyTorch

This project implements **federated learning for environmental sound classification** using **Flower**, **PyTorch**, and **torchaudio**.

It uses a federated version of the **UrbanSound8K dataset** and supports:

* **IID partitioning**
* **Natural partitioning by client (clientID)**

The system performs **audio tagging** by converting raw audio into **log-mel spectrograms** and training a compact CNN.

---

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @yan-gao/fed-audio-tagging
```

Then, install dependencies:

```bash
cd fed-audio-tagging && pip install -e .
```

Project structure:

```shell
fed-audio-tagging
├── fedaudio
│   ├── __init__.py
│   ├── client_app.py   # Client-side training
│   ├── server_app.py   # Server-side orchestration
│   └── task.py         # Model, data processing, training, evaluation
├── pyproject.toml
```

---

## Run the App

You can run this Flower App in both **simulation** and **deployment** modes.

---

### Run with the Simulation Engine

In simulation mode:

* Dataset is automatically downloaded (`flwrlabs/fed-urbansound8K`)
* Data is partitioned across clients:

  * `iid` → random split
  * `natural` → grouped by `clientID`

Run with default settings:

```bash
flwr run .
```

Override configuration:

```bash
flwr run . --run-config "num-server-rounds=20 batch-size=16"
```

Key configuration options (from `pyproject.toml`):

* `num-server-rounds`: number of FL rounds
* `local-epochs`: local training epochs
* `batch-size`: batch size
* `fraction-train`: fraction of participating clients
* `learning-rate-max/min`: cosine annealing schedule
* `partitioner`: `iid` or `natural`

---

## Model

The model is a **compact CNN for spectrogram classification**:

* Input: log-mel spectrograms
* 3 convolutional blocks with:

  * BatchNorm
  * ReLU activation
  * Max pooling
* Dropout regularization
* Global average pooling
* Fully connected classifier

Defined in: 

---

## Data Pipeline

Dataset:

* Hugging Face: `flwrlabs/fed-urbansound8K`

Processing steps:

1. Load raw audio (bytes or file path)
2. Resample to 16 kHz
3. Pad or trim to fixed length (4 seconds)
4. Convert to **mel spectrogram**
5. Convert to **log scale (dB)**
6. Normalize features

This produces input tensors of shape:

```
(batch, 1, n_mels, time)
```

Supports:

* Simulation mode via `FederatedDataset`
* Deployment mode via `load_from_disk`

Implemented in: 

---

## Training

Each client:

* Receives the global model
* Trains locally using:

  * CrossEntropyLoss
  * Adam optimizer
* Applies **cosine annealing learning rate**

Training function: 

Clients return:

* Updated model weights
* Training loss and dataset size

---

## Evaluation

Server-side evaluation:

* Uses centralized **test split**
* Reports:

  * Loss
  * Accuracy

Evaluation logic: 

---

## Run with the Deployment Engine

To run in deployment mode:

### Step 1: Prepare local datasets

Prepare audio datasets in Hugging Face format and store locally.

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

* Designed for **audio classification tasks**
* Uses **log-mel spectrograms** as features
* Handles variable-length audio via padding/trimming
* Efficient CNN architecture for edge devices
* Automatically uses GPU if available

---
