"""fed_fraud: federated financial fraud detection."""

import json
import pickle

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    Bulyan,
    Krum,
)

from fed_fraud.benchmarking import (
    BenchmarkFedAdagrad,
    BenchmarkFedAdam,
    BenchmarkFedAvg,
    BenchmarkFedAvgM,
    BenchmarkFedProx,
    BenchmarkFedYogi,
    build_communication_summary,
)
from fed_fraud.task import Net, get_input_dim, load_centralized_dataset, test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    num_rounds = int(context.run_config["num-server-rounds"])
    run_name = str(context.run_config["run-name"])
    strategy_name = context.run_config["strategy"]

    global_model = Net(
        input_dim=get_input_dim(),
        hidden_dim_1=int(context.run_config["hidden-dim-1"]),
        hidden_dim_2=int(context.run_config["hidden-dim-2"]),
        dropout=float(context.run_config["dropout"]),
    )
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

    print(f"\nSaving result to disk as result_{run_name}.pkl...")
    with open(f"result_{run_name}.pkl", "wb") as f:
        pickle.dump(result, f)
    with open(f"result_{run_name}_communication.json", "w", encoding="utf-8") as f:
        json.dump(
            build_communication_summary(
                result,
                verification_summary=getattr(strategy, "verification_summary", None),
            ),
            f,
            indent=2,
            sort_keys=True,
        )


def get_strategy(strategy_name: str, fraction_train: float, fraction_evaluate: float, context: Context):
    """Get strategy based on the strategy name."""
    benchmark_kwargs = {
        "benchmark_verify_dataset": _bool_config(
            context.run_config.get("benchmark-verify-dataset", False)
        ),
        "benchmark_manifest_path": str(
            context.run_config.get("benchmark-manifest-path", "")
        )
        or None,
    }
    if strategy_name.lower() == "fedavg":
        return BenchmarkFedAvg(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                **benchmark_kwargs,
               )
    elif strategy_name.lower() == "fedprox":
        return BenchmarkFedProx(
                fraction_train=fraction_train,
                fraction_evaluate=fraction_evaluate,
                proximal_mu=float(context.run_config["fedprox-mu"]),
                **benchmark_kwargs,
               )
    elif strategy_name.lower() == "fedavgm":
        return BenchmarkFedAvgM(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                **benchmark_kwargs,
               )
    elif strategy_name.lower() == "fedadam":
        return BenchmarkFedAdam(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
                **benchmark_kwargs,
               )   
    elif strategy_name.lower() == "fedadagrad":
        return BenchmarkFedAdagrad(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
                **benchmark_kwargs,
               )
    elif strategy_name.lower() == "fedyogi":
        return BenchmarkFedYogi(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
                **benchmark_kwargs,
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


def _bool_config(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _benchmark_config(context: Context) -> ConfigRecord:
    return ConfigRecord(
        {
            "benchmark-system-metrics": _bool_config(
                context.run_config.get("benchmark-system-metrics", False)
            )
        }
    )
    

def _maybe_global_evaluate_fn(context: Context):
    if not _bool_config(context.run_config.get("benchmark-run-server-eval", True)):
        return None
    return global_evaluate_fn(context)


def global_evaluate_fn(context: Context):
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Net(
            input_dim=get_input_dim(),
            hidden_dim_1=int(context.run_config["hidden-dim-1"]),
            hidden_dim_2=int(context.run_config["hidden-dim-2"]),
            dropout=float(context.run_config["dropout"]),
        )
        model.load_state_dict(arrays.to_torch_state_dict())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        testloader = load_centralized_dataset(
            batch_size=int(context.run_config["batch-size"])
        )
        test_loss, metrics = test(model, testloader, device)

        record = {
            "centralized_loss": float(test_loss),
        }

        for threshold, submetrics in metrics.items():
            for name, value in submetrics.items():
                record[f"{name}_at_{threshold}"] = float(value)

        return MetricRecord(record)

    return global_evaluate
