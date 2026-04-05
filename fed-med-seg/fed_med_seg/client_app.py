"""fed_med_seg: BraTS federated segmentation task."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fed_med_seg.task import Net, cosine_annealing, load_local_data, load_sim_data
from fed_med_seg.task import train as train_fn

app = ClientApp()


def _roi_size(context: Context) -> tuple[int, int, int]:
    return (
        int(context.run_config["roi-x"]),
        int(context.run_config["roi-y"]),
        int(context.run_config["roi-z"]),
    )


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local client training data only."""
    model = Net(
        in_channels=4,
        out_channels=int(context.run_config["num-classes"]),
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = int(context.run_config["batch-size"])
    roi_size = _roi_size(context)
    num_rounds = int(context.run_config["num-server-rounds"])
    learning_rate_max = float(context.run_config["learning-rate-max"])
    learning_rate_min = float(context.run_config["learning-rate-min"])


    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        partitioner_name = str(context.run_config["partitioner"])

        trainloader = load_sim_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
            roi_size=roi_size,
            partitioner_name=partitioner_name,
        )
    else:
        data_path = context.node_config["data-path"]
        trainloader = load_local_data(
            data_path=data_path,
            batch_size=batch_size,
            roi_size=roi_size,
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
