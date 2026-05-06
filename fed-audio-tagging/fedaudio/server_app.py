"""fedaudio: A Flower / PyTorch app (federated audio tagging)."""

import json
import pickle
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    Bulyan,
    Krum,
)

from fedaudio.benchmarking import (
    BenchmarkFedAdagrad,
    BenchmarkFedAdam,
    BenchmarkFedAvg,
    BenchmarkFedAvgM,
    BenchmarkFedProx,
    BenchmarkFedYogi,
    build_communication_summary,
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
        train_config=_benchmark_config(context),
        evaluate_config=_benchmark_config(context),
        evaluate_fn=_maybe_global_evaluate_fn(context),
    )

    print(f"\nSaving result to disk as result_{run_name}.pkl...")
    with open(f"result_{run_name}.pkl", "wb") as f:
        pickle.dump(result, f)
    with open(f"result_{run_name}_communication.json", "w", encoding="utf-8") as f:
        json.dump(build_communication_summary(result), f, indent=2, sort_keys=True)


def get_strategy(strategy_name: str, fraction_train: float, fraction_evaluate: float, context: Context):
    """Get strategy based on the strategy name."""
    if strategy_name.lower() == "fedavg":
        return BenchmarkFedAvg(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
               )
    elif strategy_name.lower() == "fedprox":
        return BenchmarkFedProx(
                fraction_train=fraction_train,
                fraction_evaluate=fraction_evaluate,
                proximal_mu=float(context.run_config["fedprox-mu"]),
               )
    elif strategy_name.lower() == "fedavgm":
        return BenchmarkFedAvgM(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
               )
    elif strategy_name.lower() == "fedadam":
        return BenchmarkFedAdam(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
               )   
    elif strategy_name.lower() == "fedadagrad":
        return BenchmarkFedAdagrad(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
               )
    elif strategy_name.lower() == "fedyogi":
        return BenchmarkFedYogi(
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


def _benchmark_config(context: Context) -> ConfigRecord:
    return ConfigRecord(
        {
            "benchmark-system-metrics": _bool_config(
                context.run_config.get("benchmark-system-metrics", False)
            )
        }
    )


def _bool_config(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _maybe_global_evaluate_fn(context: Context):
    if not _bool_config(context.run_config.get("benchmark-run-server-eval", True)):
        return None
    return global_evaluate_fn(context)


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
