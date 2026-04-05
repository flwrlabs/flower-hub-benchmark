"""fedaudio: A Flower / PyTorch app (federated audio tagging)."""

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

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        float(new_lr),
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
