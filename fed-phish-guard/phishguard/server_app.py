"""fed-phish-guard: A Flower / PyTorch app (federated phishing URL detection)."""

from __future__ import annotations

import pickle
import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

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

    global_model = PhishingCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        num_filters=num_filters,
        dropout=dropout,
    )
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        fraction_train=fraction_train,
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
