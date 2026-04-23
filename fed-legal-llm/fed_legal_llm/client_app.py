"""fed_legal_llm client app."""

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

    train_loss = train_fn(
        model=model,
        trainloader=trainloader,
        epochs=int(context.run_config["local-epochs"]),
        lr=float(new_lr),
        weight_decay=weight_decay,
        device=device,
        show_progress=bool(context.run_config.get("show-progress", False)),
    )

    lora_state = model.get_lora_state_dict()
    num_examples = len(getattr(trainloader, "dataset", []))
    content = RecordDict(
        {
            "arrays": ArrayRecord(lora_state),
            "metrics": MetricRecord(
                {
                    "train_loss": float(train_loss),
                    "num-examples": int(num_examples),
                    "learning_rate": float(new_lr),
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)
