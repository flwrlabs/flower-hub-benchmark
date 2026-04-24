"""fed-phish-guard: A Flower / PyTorch app (federated phishing URL detection)."""

from __future__ import annotations

import json
import pickle
import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    Bulyan,
    Krum,
)

from phishguard.benchmarking import (
    BenchmarkFedAdagrad,
    BenchmarkFedAdam,
    BenchmarkFedAvg,
    BenchmarkFedAvgM,
    BenchmarkFedProx,
    BenchmarkFedYogi,
    build_communication_summary,
)
from phishguard.data import VOCAB_SIZE, load_centralized_dataset
from phishguard.model import PhishingCNN
from phishguard.train import evaluate as test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    fraction_train: float = float(context.run_config["fraction-train"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    num_rounds: int = int(context.run_config["num-server-rounds"])
    embed_dim = int(context.run_config["embed-dim"])
    num_filters = int(context.run_config["num-filters"])
    dropout = float(context.run_config["dropout"])
    run_name = str(context.run_config["run-name"])
    strategy_name = context.run_config["strategy"]

    global_model = PhishingCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        num_filters=num_filters,
        dropout=dropout,
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
        evaluate_fn=global_evaluate_fn(context),
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
    

def global_evaluate_fn(context: Context):
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = PhishingCNN(
            vocab_size=VOCAB_SIZE,
            embed_dim=int(context.run_config["embed-dim"]),
            num_filters=int(context.run_config["num-filters"]),
            dropout=float(context.run_config["dropout"]),
        )
        model.load_state_dict(arrays.to_torch_state_dict())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        testloader = load_centralized_dataset(
            batch_size=int(context.run_config["batch-size"]),
        )

        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
        test_metrics, _, _ = test(model, testloader, pos_weight, device)

        return MetricRecord(
            {
                "centralized_loss": float(test_metrics["loss"]),
                "centralized_accuracy": float(test_metrics["accuracy"]),
                "centralized_precision": float(test_metrics["precision"]),
                "centralized_recall": float(test_metrics["recall"]),
                "centralized_f1": float(test_metrics["f1"]),
                "centralized_auc": float(test_metrics["auc"]),
            }
        )

    return global_evaluate
