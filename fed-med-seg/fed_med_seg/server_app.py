"""fed_med_seg: BraTS federated segmentation task."""

import pickle
import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fed_med_seg.task import Net, load_centralized_dataset, test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    num_rounds = int(context.run_config["num-server-rounds"])
    num_classes = int(context.run_config["num-classes"])
    run_name = context.run_config["run-name"]

    global_model = Net(in_channels=4, out_channels=num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate_fn(context),
    )

    # Save result
    print(f"\nSaving result to disk as result_{run_name}.pkl...")
    with open(f"result_{run_name}.pkl", "wb") as f:
        pickle.dump(result, f)


def global_evaluate_fn(context: Context):
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Net(
            in_channels=4,
            out_channels=int(context.run_config["num-classes"]),
        )
        model.load_state_dict(arrays.to_torch_state_dict())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        testloader = load_centralized_dataset(batch_size=context.run_config["batch-size"])
        test_loss, test_dice = test(
            model,
            testloader,
            device,
            num_classes=int(context.run_config["num-classes"]),
        )

        return MetricRecord(
            {
                "centralized_loss": float(test_loss),
                "centralized_dice": float(test_dice),
            }
        )

    return global_evaluate
