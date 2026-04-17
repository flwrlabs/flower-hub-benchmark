"""fedaudio: A Flower / PyTorch app (federated audio tagging)."""

import pickle
from pyexpat import model
import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    Bulyan,
    FedAdagrad,
    FedAdam, 
    FedAvg,
    FedAvgM, 
    FedProx,
    FedYogi,
    Krum,
)

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
    strategy_name = context.run_config["strategy"]

    # Load global model
    global_model = make_model()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize strategy
    strategy = get_strategy(
        strategy_name=strategy_name,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        context=context,
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


def get_strategy(strategy_name: str, fraction_train: float, fraction_evaluate: float, context: Context):
    """Get strategy based on the strategy name."""
    if strategy_name.lower() == "fedavg":
        return FedAvg(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
               )
    elif strategy_name.lower() == "fedprox":
        return FedProx(
                fraction_train=fraction_train,
                fraction_evaluate=fraction_evaluate,
                proximal_mu=float(context.run_config["fedprox-mu"]),
               )
    elif strategy_name.lower() == "fedavgm":
        return FedAvgM(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
               )
    elif strategy_name.lower() == "fedadam":
        return FedAdam(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
               )   
    elif strategy_name.lower() == "fedadagrad":
        return FedAdagrad(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
               )
    elif strategy_name.lower() == "fedyogi":
        return FedYogi(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
               )
    elif strategy_name.lower() == "krum":
        return Krum(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
               )
    elif strategy_name.lower() == "bulyan":
        return Bulyan(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
               )
    else:
        raise ValueError(f"Unsupported strategy '{strategy_name}'.")


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
