"""fedaudio: A Flower / PyTorch app (federated audio tagging)."""

import pickle
from pyexpat import model
import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedaudio.task import load_centralized_dataset, make_model, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    run_name = str(context.run_config["run-name"])

    # Load global model
    global_model = make_model()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        fraction_train=fraction_train,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate_fn(context),
    )

    print(f"\nSaving result to disk as result_{run_name}.pkl...")
    with open(f"result_{run_name}.pkl", "wb") as f:
        pickle.dump(result, f)


def global_evaluate_fn(context: Context):
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = make_model()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        testloader = load_centralized_dataset(
            batch_size=int(context.run_config["batch-size"])
        )
        test_loss, test_acc = test(model, testloader, device)

        return MetricRecord(
            {
                "eval_loss": float(test_loss),
                "eval_acc": float(test_acc),
            }
        )

    return global_evaluate
