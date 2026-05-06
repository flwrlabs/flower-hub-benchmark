---
tags: [benchmark, llm, federated-learning, lora, legal-ai]
dataset: [fed-legal]
framework: [torch, transformers, peft, flwr]
---

# Federated Legal LLM Fine-Tuning with Flower and LoRA

This example demonstrates how to perform **federated fine-tuning of a large language model (LLM)** using **Flower**, **PyTorch**, **Transformers**, and **PEFT/LoRA**. It uses an **English-only 5-silo legal instruction-tuning dataset** hosted on Hugging Face and supports both **simulation** and **deployment** workflows.

The project includes:

* A **causal language model** for instruction tuning
* **LoRA-based parameter-efficient fine-tuning**
* Data loading pipelines using **Hugging Face Datasets**
* Federated training with **Flower**
* Support for both **simulation** and **deployment** workflows

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/fed-legal-llm
```

Then, install dependencies:

```bash
cd fed-legal-llm && pip install -e .
```

Project structure:

```shell
fed-legal-llm
├── fed_legal_llm
│   ├── __init__.py
│   ├── client_app.py   # Client-side LoRA fine-tuning logic
│   ├── server_app.py   # Server-side orchestration and evaluation
│   └── task.py         # Model, data loading, training, evaluation
├── pyproject.toml      # Dependencies and configuration
└── README.md
```

## Run the App

This Flower App supports both **simulation mode** and **deployment mode** without code changes.

### Run with the Simulation Engine

In simulation mode:

* The **federated legal dataset** is automatically downloaded from Hugging Face
* Data is partitioned across clients using the dataset’s natural client identifier
* Each client trains on its own legal silo using **LoRA fine-tuning**

Run with default configuration:

```bash
flwr run .
```

Override configuration (example):

```bash
flwr run . --run-config "num-server-rounds=5 local-epochs=1 batch-size=2"
```

Key configurable parameters (from `pyproject.toml`):

* `model-name`: pretrained base model to fine-tune
* `dataset-id`: Hugging Face dataset repo
* `num-server-rounds`: number of FL rounds
* `local-epochs`: local training epochs per client
* `batch-size`: local batch size
* `learning-rate`: local optimizer learning rate
* `max-seq-length`: maximum tokenized sequence length
* `load-in-4bit`: whether to use 4-bit quantized loading
* `torch-dtype`: model compute dtype
* `lora-r`: LoRA rank
* `lora-alpha`: LoRA alpha
* `lora-dropout`: LoRA dropout
* `lora-target-modules`: modules to adapt with LoRA
* `fraction-train`: fraction of clients sampled for training each round
* `fraction-evaluate`: fraction of clients sampled for client-side evaluation if enabled

### Model

The default model is:

* `HuggingFaceTB/SmolLM3-3B`

The code is written to make switching to another pre-trained causal LLM easy by changing the model name in `pyproject.toml`, for example:

* `mistralai/Mistral-7B-Instruct-v0.3`

The model is loaded with Hugging Face `AutoModelForCausalLM` and tokenized with `AutoTokenizer`. LoRA adapters are applied using PEFT, so only the **adapter parameters** are trained and federated.

Typical LoRA configuration includes:

* `r`
* `alpha`
* `dropout`
* `target_modules`

This keeps communication and GPU memory requirements much lower than full-model fine-tuning.

### Dataset

Data is loaded from a Hugging Face dataset containing three splits:

* `train`
* `valid`
* `test`

Each row contains:

* `messages`
* `client_id`
* `task_type`
* `language`
* `example_id`

The dataset is an English-only **5-silo legal instruction dataset** with the following clients:

* `0`: legal provision classification
* `1`: case holding selection
* `2`: unfair Terms of Service classification
* `3`: Supreme Court issue classification
* `4`: contract NLI / legal reasoning

In simulation mode:

* clients are identified using the dataset’s `client_id`
* each Flower client trains only on its own silo

In deployment mode:

* each node loads a pre-partitioned local dataset from disk

### Data Pipeline

The data pipeline:

* loads the Hugging Face dataset
* filters rows by `client_id`
* applies the tokenizer’s chat template to `messages`
* tokenizes the prompt/response sequence
* creates labels for supervised causal language modeling

Two modes are supported:

* **Simulation mode** → partitions the centralized Hugging Face dataset by `client_id`
* **Deployment mode** → loads locally stored partitions with `load_from_disk(...)`

The code is designed so that changing the underlying base model only requires updating configuration, as long as the model is compatible with Hugging Face `transformers`.

### Training

Each client:

* receives the current global LoRA adapter weights
* reconstructs the base model and applies the LoRA adapters
* trains locally on its own legal training partition
* returns updated adapter weights to the server

Local training uses:

* causal language modeling loss
* AdamW optimizer
* LoRA fine-tuning through PEFT

Only the LoRA weights are exchanged during federation, not the full base model weights.

This makes the setup practical for large instruction-tuned models.

Training logic is defined in:

* `fed_legal_llm/task.py`
* `fed_legal_llm/client_app.py`

### Evaluation

Server-side evaluation:

* uses the centralized **test split**
* evaluates the global model after each federated round
* reports text-generation metrics suitable for all legal silos

The evaluation metrics are:

* **Exact Match (EM)** — strict normalized match between prediction and reference
* **Token-level F1** — overlap-based score between prediction and reference
* **Macro client F1** — average token-level F1 across clients, giving equal weight to each client
* **Macro client EM** — average exact match across clients

Why these metrics:

* the dataset contains heterogeneous legal tasks
* all tasks are cast as text generation
* exact match is strict and interpretable
* token F1 gives partial credit when answers are close but not identical
* macro client scores prevent large silos from dominating the overall result

### Run with the Deployment Engine

For deployment, you must provide local dataset partitions.

#### Step 1: Prepare data

Download and partition the dataset manually, then store one local partition per node.

Each local partition should contain examples from only one `client_id`, or otherwise represent that node’s local legal data.

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

## Example Configuration

A typical setup in `pyproject.toml` may look like this:

```toml
model-name = "HuggingFaceTB/SmolLM3-3B"
dataset-id = "flwrlabs/fed-legal"
load-in-4bit = true
torch-dtype = "bfloat16"

lora-r = 16
lora-alpha = 32
lora-dropout = 0.05
lora-target-modules = "q_proj,k_proj,v_proj,o_proj"

num-server-rounds = 5
local-epochs = 1
batch-size = 2
learning-rate = 2e-4
max-seq-length = 2048
fraction-train = 1.0
fraction-evaluate = 1.0
```

To switch models, update:

```toml
model-name = "mistralai/Mistral-7B-Instruct-v0.3"
```

and, if needed, adjust LoRA target modules and dtype.
