"""fed_med_seg: BraTS federated segmentation task."""

import json
import pickle
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    Bulyan,
    Krum,
)

from fed_med_seg.benchmarking import (
    BenchmarkFedAdagrad,
    BenchmarkFedAdam,
    BenchmarkFedAvg,
    BenchmarkFedAvgM,
    BenchmarkFedProx,
    BenchmarkFedYogi,
    build_communication_summary,
)
from fed_med_seg.task import Net, load_centralized_dataset, test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    num_rounds = int(context.run_config["num-server-rounds"])
    num_classes = int(context.run_config["num-classes"])
    run_name = context.run_config["run-name"]
    strategy_name = context.run_config["strategy"]

    global_model = Net(in_channels=4, out_channels=num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    strategy = get_strategy(
        strategy_name=strategy_name, 
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        context=context,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        train_config=_benchmark_config(context),
        evaluate_config=_benchmark_config(context),
        evaluate_fn=_maybe_global_evaluate_fn(context),
    )

    # Save result
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
               )   
    elif strategy_name.lower() == "fedadagrad":
        return BenchmarkFedAdagrad(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
               )
    elif strategy_name.lower() == "fedyogi":
        return BenchmarkFedYogi(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
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
