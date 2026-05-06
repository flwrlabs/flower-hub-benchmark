"""fed_fraud: federated financial fraud detection."""

import resource
import sys
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fed_fraud.task import (
    Net,
    cosine_annealing,
    get_input_dim,
    load_local_data,
    load_local_metadata,
    load_sim_data,
    load_sim_metadata,
)
from fed_fraud.task import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local client training data only."""
    benchmark_mode = str(
        msg.content["config"].get("benchmark_mode", "train")
    ).strip().lower()
    if benchmark_mode == "verify_only":
        content = RecordDict({"config": ConfigRecord(_load_metadata(context))})
        return Message(content=content, reply_to=msg)

    model = Net(
        input_dim=get_input_dim(),
        hidden_dim_1=int(context.run_config["hidden-dim-1"]),
        hidden_dim_2=int(context.run_config["hidden-dim-2"]),
        dropout=float(context.run_config["dropout"]),
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    system_metrics_enabled = _system_metrics_enabled(context)

    batch_size = int(context.run_config["batch-size"])
    num_rounds = int(context.run_config["num-server-rounds"])
    learning_rate_max = float(context.run_config["learning-rate-max"])
    learning_rate_min = float(context.run_config["learning-rate-min"])
    use_class_weights = str(context.run_config["use-class-weights"]).lower() == "true"

    if "partition-id" in context.node_config and "num-partitions" in context.node_config:
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        partitioner_name = str(context.run_config["partitioner"])

        trainloader = load_sim_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
            partitioner_name=partitioner_name,
        )
    else:
        data_path = context.node_config["data-path"]
        trainloader = load_local_data(
            data_path=data_path,
            batch_size=batch_size,
            split="train",
        )

    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
        num_rounds,
        learning_rate_max,
        learning_rate_min,
    )

    if system_metrics_enabled and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter() if system_metrics_enabled else 0.0
    train_loss = train_fn(
        model,
        trainloader,
        int(context.run_config["local-epochs"]),
        float(new_lr),
        device,
        use_class_weights=use_class_weights,
    )
    client_train_time_sec = (
        time.perf_counter() - train_start if system_metrics_enabled else 0.0
    )

    metrics = {
        "train_loss": float(train_loss),
        "num-examples": len(trainloader.dataset),
        "learning_rate": float(new_lr),
    }
    if system_metrics_enabled:
        metrics["client_train_time_sec"] = float(client_train_time_sec)
        metrics["client_peak_cpu_memory_mb"] = float(_peak_rss_mb())
        metrics["client_peak_gpu_memory_mb"] = float(_peak_gpu_memory_mb(device))

    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=content, reply_to=msg)


def _load_metadata(context: Context) -> dict[str, int | str]:
    """Select simulation or deployment metadata based on node_config."""
    if "partition-id" in context.node_config and "num-partitions" in context.node_config:
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
