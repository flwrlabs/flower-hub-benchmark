"""fedaudio: A Flower / PyTorch app (federated audio tagging)."""

import resource
import sys
import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedaudio.task import cosine_annealing, load_local_data, load_sim_data, make_model
from fedaudio.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    system_metrics_enabled = _system_metrics_enabled(context)

    # Load the data
    batch_size = context.run_config["batch-size"]
    num_rounds = int(context.run_config["num-server-rounds"])
    learning_rate_max = float(context.run_config["learning-rate-max"])
    learning_rate_min = float(context.run_config["learning-rate-min"])
    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        # Simulation engine: use `flwr_datasets` and partition data on the fly
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        partitioner_name = str(context.run_config["partitioner"])
        trainloader = load_sim_data(partition_id, num_partitions, batch_size, partitioner_name)
    else:
        # Deployment engine: load demo data or real user data
        data_path = context.node_config["data-path"]
        trainloader = load_local_data(data_path, batch_size)
    
    # Compute the new learning rate using cosine annealing
    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
        num_rounds,
        learning_rate_max,
        learning_rate_min,
    )

    if system_metrics_enabled and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter() if system_metrics_enabled else 0.0
    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        float(new_lr),
        device,
    )
    client_train_time_sec = (
        time.perf_counter() - train_start if system_metrics_enabled else 0.0
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "learning_rate": float(new_lr),
    }
    if system_metrics_enabled:
        metrics["client_train_time_sec"] = float(client_train_time_sec)
        metrics["client_peak_cpu_memory_mb"] = float(_peak_rss_mb())
        metrics["client_peak_gpu_memory_mb"] = float(_peak_gpu_memory_mb(device))
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
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
