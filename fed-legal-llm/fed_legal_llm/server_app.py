"""fed_legal_llm server app."""

import json
import pickle

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Bulyan, Krum

from fed_legal_llm.benchmarking import (
    BenchmarkFedAdagrad,
    BenchmarkFedAdam,
    BenchmarkFedAvg,
    BenchmarkFedAvgM,
    BenchmarkFedProx,
    BenchmarkFedYogi,
    build_communication_summary,
)
from fed_legal_llm.task import evaluate_generation, load_centralized_testset, load_model_from_config

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    num_rounds = int(context.run_config["num-server-rounds"])
    run_name = str(context.run_config["run-name"])
    strategy_name = str(context.run_config["strategy"])

    global_model = load_model_from_config(context)
    arrays = ArrayRecord(global_model.get_lora_state_dict())

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
        evaluate_fn=_maybe_global_evaluate_fn(context),
        train_config=_benchmark_config(context),
        evaluate_config=_benchmark_config(context),
        timeout=10800,  # 3 hours
    )

    with open(f"result_{run_name}.pkl", "wb") as f:
        pickle.dump(result, f)
    with open(f"result_{run_name}_communication.json", "w", encoding="utf-8") as f:
        json.dump(build_communication_summary(result), f, indent=2, sort_keys=True)


def get_strategy(strategy_name: str, fraction_train: float, fraction_evaluate: float, context: Context):
    name = strategy_name.lower()
    if name == "fedavg":
        return BenchmarkFedAvg(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    if name == "fedprox":
        return BenchmarkFedProx(
                fraction_train=fraction_train,
                fraction_evaluate=fraction_evaluate,
                proximal_mu=float(context.run_config["fedprox-mu"]),
               )
    if name == "fedavgm":
        return BenchmarkFedAvgM(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    if name == "fedadam":
        return BenchmarkFedAdam(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
               )
    if name == "fedadagrad":
        return BenchmarkFedAdagrad(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
               )
    if name == "fedyogi":
        return BenchmarkFedYogi(
                fraction_train=fraction_train, 
                fraction_evaluate=fraction_evaluate,
                eta=float(context.run_config["eta"]),
                eta_l=float(context.run_config["eta_l"]),
               )
    if name == "krum":
        return Krum(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    if name == "bulyan":
        return Bulyan(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
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
        model = load_model_from_config(context)
        model.set_lora_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset_id = str(context.run_config["dataset-id"])
        test_ds = load_centralized_testset(dataset_id=dataset_id)
        metrics = evaluate_generation(
            model=model,
            dataset=test_ds,
            device=device,
            max_new_tokens=int(context.run_config["eval-max-new-tokens"]),
            eval_batch_size=int(context.run_config["eval-batch-size"]),
        )
        return MetricRecord({k: float(v) for k, v in metrics.items()})

    return global_evaluate
