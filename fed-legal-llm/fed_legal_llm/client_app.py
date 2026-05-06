"""fed_legal_llm client app."""

import resource
import sys
import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fed_legal_llm.task import (
    cosine_annealing,
    load_local_data,
    load_model_from_config,
    load_sim_data,
    train as train_fn,
)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train LoRA adapter weights on one client's local split."""
    model = load_model_from_config(context)
    incoming = msg.content["arrays"].to_torch_state_dict()
    model.set_lora_state_dict(incoming)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    system_metrics_enabled = _system_metrics_enabled(context)
    batch_size = int(context.run_config["batch-size"])
    num_rounds = int(context.run_config["num-server-rounds"])
    learning_rate_max = float(context.run_config["learning-rate-max"])
    learning_rate_min = float(context.run_config["learning-rate-min"])
    weight_decay = float(context.run_config["weight-decay"])
    max_length = int(context.run_config["max-length"])
    dataset_id = str(context.run_config["dataset-id"])

    if "partition-id" in context.node_config and "num-partitions" in context.node_config:
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        partitioner_name = str(context.run_config["partitioner"])
        trainloader = load_sim_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
            tokenizer=model.tokenizer,
            max_length=max_length,
            partitioner_name=partitioner_name,
            dataset_id=dataset_id,
        )
    else:
        data_path = str(context.node_config["data-path"])
        trainloader = load_local_data(
            data_path=data_path,
            batch_size=batch_size,
            tokenizer=model.tokenizer,
            max_length=max_length,
        )

    new_lr = cosine_annealing(
        int(msg.content["config"]["server-round"]),
        num_rounds,
        learning_rate_max,
        learning_rate_min,
    )

    if system_metrics_enabled and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter() if system_metrics_enabled else 0.0
    train_loss = train_fn(
        model=model,
        trainloader=trainloader,
        epochs=int(context.run_config["local-epochs"]),
        lr=float(new_lr),
        weight_decay=weight_decay,
        device=device,
        show_progress=bool(context.run_config.get("show-progress", False)),
    )
    client_train_time_sec = (
        time.perf_counter() - train_start if system_metrics_enabled else 0.0
    )

    lora_state = model.get_lora_state_dict()
    num_examples = len(getattr(trainloader, "dataset", []))
    metrics = {
        "train_loss": float(train_loss),
        "num-examples": int(num_examples),
        "learning_rate": float(new_lr),
    }
    if system_metrics_enabled:
        metrics["client_train_time_sec"] = float(client_train_time_sec)
        metrics["client_peak_cpu_memory_mb"] = float(_peak_rss_mb())
        metrics["client_peak_gpu_memory_mb"] = float(_peak_gpu_memory_mb(device))
    content = RecordDict(
        {
            "arrays": ArrayRecord(lora_state),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=content, reply_to=msg)


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
