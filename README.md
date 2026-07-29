# Flower Hub Benchmark

This repository contains five realistic federated learning (FL) benchmark
applications. They cover cross-silo and cross-device settings and span medical
images, financial tabular data, legal text, URLs, and audio.

| Application | Task | Default rounds | Natural partition |
|---|---|---:|---|
| [`fed-med-seg`](./fed-med-seg/) | Brain-tumour segmentation | 20 | Institution/source silo |
| [`fed-fin-fraud`](./fed-fin-fraud/) | Financial fraud detection | 20 | Bank |
| [`fed-legal-llm`](./fed-legal-llm/) | Legal instruction tuning | 10 | Legal task silo |
| [`fed-phish-guard`](./fed-phish-guard/) | Phishing URL detection | 20 | Dataset client ID |
| [`fed-audio-tagging`](./fed-audio-tagging/) | Urban audio tagging | 100 | Audio client ID |

Each application has its own README and `pyproject.toml`. The applications are
also available on [Flower Hub](https://flower.ai/apps/):
[`fed-med-seg`](https://flower.ai/apps/flwrlabs/fed-med-seg/),
[`fed-fin-fraud`](https://flower.ai/apps/flwrlabs/fed-fin-fraud/),
[`fed-legal-llm`](https://flower.ai/apps/flwrlabs/fed-legal-llm/),
[`fed-phish-guard`](https://flower.ai/apps/flwrlabs/fed-phish-guard/), and
[`fed-audio-tagging`](https://flower.ai/apps/flwrlabs/fed-audio-tagging/).

## Install all benchmark dependencies

The root [`requirements.txt`](./requirements.txt) provides one reconciled
environment for running the complete benchmark sweep:

```bash
# Use a Python version supported by Flower and PyTorch (for example, 3.11).
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The individual application manifests contain conflicting historical pins
(including different PyTorch and Hugging Face Datasets versions). The root
requirements select one compatible stack for cross-application sweeps. Use an
application's own `pyproject.toml` instead when creating an isolated environment
for only that application.

GPU-specific PyTorch installations may require the appropriate package index
for the machine's CUDA version. The legal LLM uses 4-bit loading by default,
which requires a supported accelerator and `bitsandbytes`. On macOS, disable
4-bit loading through the sweep's extra run configuration.

## Sweep every strategy over every task

The root [`sweep_all_strategies.sh`](./sweep_all_strategies.sh) runs these six
strategies:

* FedAvg
* FedProx
* FedAvgM
* FedAdam
* FedAdagrad
* FedYogi

across all five applications, producing 30 sequential runs:

```bash
./sweep_all_strategies.sh
```

The script always passes `partitioner=natural`; IID runs are intentionally not
included. Apart from `strategy`, `run-name`, and the natural partitioner, every
application keeps the task-specific defaults in its `pyproject.toml`. This
includes its number of rounds and strategy hyperparameters, so the sweep does
not imply that one common optimizer configuration is optimal for every task.

A complete sweep is computationally expensive, especially for 3D medical
segmentation and the 3B-parameter legal LLM. Inspect commands before launching:

```bash
./sweep_all_strategies.sh --dry-run
```

Run a short subset:

```bash
./sweep_all_strategies.sh \
  --tasks fed-fin-fraud,fed-phish-guard \
  --strategies fedavg,fedyogi \
  --rounds 2
```

Pass an explicit Flower SuperLink connection and federation:

```bash
./sweep_all_strategies.sh \
  --superlink my-superlink \
  --federation @account/my-federation
```

Pass additional run configuration to every selected experiment:

```bash
./sweep_all_strategies.sh \
  --extra-run-config "benchmark-run-server-eval=false"
```

For a legal-LLM run on a platform without `bitsandbytes`, for example:

```bash
./sweep_all_strategies.sh \
  --tasks fed-legal-llm \
  --extra-run-config "load-in-4bit=false torch-dtype=float32"
```

Use `./sweep_all_strategies.sh --help` for all options.

### Sweep outputs

Outputs are written beneath:

```text
sweep-results/<sweep-id>/
├── summary.tsv
└── <task>/<strategy>/
    ├── run.log
    ├── result_<run-name>.pkl                # local runs, when produced
    └── result_<run-name>_communication.json # local runs, when produced
```

The sweep continues after an individual failure and records each exit code in
`summary.tsv`. It exits with status 1 if any run failed. Pass `--fail-fast` to
stop at the first failure. Logs and the summary are always local. When using a
remote SuperLink, result artifacts written by a ServerApp remain on the remote
runtime and therefore cannot be moved into the local sweep directory.
