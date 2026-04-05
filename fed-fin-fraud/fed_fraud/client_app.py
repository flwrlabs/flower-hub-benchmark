"""fed_fraud: federated financial fraud detection."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fed_fraud.task import (
    Net,
    cosine_annealing,
    get_input_dim,
    load_local_data,
    load_sim_data,
)
from fed_fraud.task import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local client training data only."""
    model = Net(
        input_dim=get_input_dim(),
        hidden_dim_1=int(context.run_config["hidden-dim-1"]),
        hidden_dim_2=int(context.run_config["hidden-dim-2"]),
        dropout=float(context.run_config["dropout"]),
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    train_loss = train_fn(
        model,
        trainloader,
        int(context.run_config["local-epochs"]),
        float(new_lr),
        device,
        use_class_weights=use_class_weights,
    )

    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "train_loss": float(train_loss),
                    "num-examples": len(trainloader.dataset),
                    "learning_rate": float(new_lr),
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)
