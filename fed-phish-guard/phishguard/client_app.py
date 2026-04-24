"""fed-phish-guard: A Flower / PyTorch app (federated phishing URL detection)."""

from __future__ import annotations

import resource
import sys
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from phishguard.data import (
    VOCAB_SIZE,
    load_local_data,
    load_local_metadata,
    load_sim_data,
    load_sim_metadata,
)
from phishguard.model import PhishingCNN
from phishguard.train import cosine_annealing, summarize_history
from phishguard.train import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    benchmark_mode = str(
        msg.content["config"].get("benchmark_mode", "train")
    ).strip().lower()
    if benchmark_mode == "verify_only":
        metadata = _load_metadata(context)
        config_record = ConfigRecord(metadata)
        content = RecordDict({"config": config_record})
        return Message(content=content, reply_to=msg)

    model, device = _load_model(msg, context)
    system_metrics_enabled = _system_metrics_enabled(context)
    batch_size = int(context.run_config["batch-size"])
    num_rounds = int(context.run_config["num-server-rounds"])
    learning_rate_max = float(context.run_config["learning-rate-max"])
    learning_rate_min = float(context.run_config["learning-rate-min"])
    trainloader, pos_weight = _load_data(context, batch_size, device)

    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
        num_rounds,
        learning_rate_max,
        learning_rate_min,
    )

    if system_metrics_enabled and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter() if system_metrics_enabled else 0.0
    history, _ = train_fn(
        model=model,
        train_loader=trainloader,
        pos_weight=pos_weight,
        lr=float(new_lr),
        device=device,
        num_epochs=int(context.run_config["local-epochs"]),
    )
    client_train_time_sec = (
        time.perf_counter() - train_start if system_metrics_enabled else 0.0
    )

    summary = summarize_history(history)
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": float(summary["avg_train_loss"]),
        "train_accuracy": float(summary["avg_train_accuracy"]),
        "train_f1": float(summary["avg_train_f1"]),
        "num-examples": len(trainloader.dataset),
    }
    if system_metrics_enabled:
        metrics["client_train_time_sec"] = float(client_train_time_sec)
        metrics["client_peak_cpu_memory_mb"] = float(_peak_rss_mb())
        metrics["client_peak_gpu_memory_mb"] = float(_peak_gpu_memory_mb(device))
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _load_model(msg: Message, context: Context) -> tuple[PhishingCNN, torch.device]:
    """Construct model from run config and load weights from the received message."""
    model = PhishingCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=context.run_config["embed-dim"],
        num_filters=context.run_config["num-filters"],
        dropout=context.run_config["dropout"],
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def _load_data(context: Context, batch_size: int, device: torch.device):
    """Select simulation or deployment data loader based on node_config."""
    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        return load_sim_data(
            partition_id=int(context.node_config["partition-id"]),
            num_partitions=int(context.node_config["num-partitions"]),
            batch_size=batch_size,
            device=device,
            partitioner_name=str(context.run_config["partitioner"]),
        )

    if "data-path" not in context.node_config:
        raise ValueError(
            "Deployment mode requires node_config['data-path'] to point to a local dataset."
        )

    return load_local_data(
        data_path=str(context.node_config["data-path"]),
        batch_size=batch_size,
        device=device,
    )


def _load_metadata(context: Context) -> dict[str, int | str]:
    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        return load_sim_metadata(
            partition_id=int(context.node_config["partition-id"]),
            num_partitions=int(context.node_config["num-partitions"]),
            partitioner_name=str(context.run_config["partitioner"]),
        )

    if "data-path" not in context.node_config:
        raise ValueError(
            "Deployment mode requires node_config['data-path'] to point to a local dataset."
        )

    return load_local_metadata(
        data_path=str(context.node_config["data-path"]),
        dataset_version=str(context.node_config.get("dataset-version", "local")),
        fallback_partition_id=context.node_config.get("partition-id", "unknown"),
    )


# memory cost tracking methods
def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return usage / scale


def _peak_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _system_metrics_enabled(context: Context) -> bool:
    raw_value = context.node_config.get(
        "benchmark-system-metrics",
        context.run_config.get("benchmark-system-metrics", False),
    )
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
