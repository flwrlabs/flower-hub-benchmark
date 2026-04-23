"""fed_legal_llm server app."""

import pickle

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Bulyan, FedAdagrad, FedAdam, FedAvg, FedAvgM, FedProx, FedYogi, Krum

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
        evaluate_fn=global_evaluate_fn(context),
    )

    with open(f"result_{run_name}.pkl", "wb") as f:
        pickle.dump(result, f)


def get_strategy(strategy_name: str, fraction_train: float, fraction_evaluate: float, context: Context):
    name = strategy_name.lower()
    if name == "fedavg":
        return FedAvg(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    if name == "fedprox":
        return FedProx(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            proximal_mu=float(context.run_config["fedprox-mu"]),
        )
    if name == "fedavgm":
        return FedAvgM(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    if name == "fedadam":
        return FedAdam(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    if name == "fedadagrad":
        return FedAdagrad(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    if name == "fedyogi":
        return FedYogi(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    if name == "krum":
        return Krum(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    if name == "bulyan":
        return Bulyan(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    raise ValueError(f"Unsupported strategy '{strategy_name}'.")


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
        metrics["server_round"] = float(server_round)
        return MetricRecord({k: float(v) for k, v in metrics.items()})

    return global_evaluate
