"""fed_fraud: federated financial fraud detection."""

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, NaturalIdPartitioner
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


HF_DATASET_ID = "flwrlabs/fed-fraud-paysim-banks"

# Raw columns expected in the uploaded dataset
NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

CATEGORICAL_TYPE_VALUES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

LABEL_COL = "isFraud"
PARTITION_COL = "BankID"

fds = None
_preprocessor = None


@dataclass
class Preprocessor:
    means: dict[str, float]
    stds: dict[str, float]
    type_to_idx: dict[str, int]
    input_dim: int


class FraudDataset(Dataset):
    """Torch dataset wrapping preprocessed tabular fraud rows."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class Net(nn.Module):
    """MLP for fraud detection with LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout: float = 0.2,
        use_layernorm: bool = True,
    ):
        super().__init__()

        layers = []

        # First block
        layers.append(nn.Linear(input_dim, hidden_dim_1))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim_1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Second block
        layers.append(nn.Linear(hidden_dim_1, hidden_dim_2))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim_2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim_2, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _get_all_type_values(ds: HFDataset) -> list[str]:
    observed = set()
    for row in ds:
        observed.add(str(row["type"]).strip())
    merged = sorted(set(CATEGORICAL_TYPE_VALUES).union(observed))
    return merged


def _ensure_preprocessor() -> Preprocessor:
    global _preprocessor

    if _preprocessor is not None:
        return _preprocessor

    train_ds = load_dataset(HF_DATASET_ID, split="train")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for col in NUMERIC_COLUMNS:
        values = np.asarray(train_ds[col], dtype=np.float64)
        means[col] = float(values.mean())
        std = float(values.std())
        stds[col] = std if std > 1e-12 else 1.0

    type_values = _get_all_type_values(train_ds)
    type_to_idx = {t: i for i, t in enumerate(type_values)}

    # base numeric features
    # + engineered balance delta features (4)
    # + one-hot transaction type
    input_dim = len(NUMERIC_COLUMNS) + 4 + len(type_values)

    _preprocessor = Preprocessor(
        means=means,
        stds=stds,
        type_to_idx=type_to_idx,
        input_dim=input_dim,
    )
    return _preprocessor


def get_input_dim() -> int:
    return _ensure_preprocessor().input_dim


def _make_partitioner(partitioner_name: str, num_partitions: int):
    name = partitioner_name.strip().lower()

    if name == "iid":
        return IidPartitioner(num_partitions=num_partitions)

    if name == "natural":
        return NaturalIdPartitioner(partition_by=PARTITION_COL)

    raise ValueError(
        f"Unsupported partitioner '{partitioner_name}'. "
        "Use 'iid' or 'natural'."
    )


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _encode_row(row: dict[str, Any], prep: Preprocessor) -> np.ndarray:
    feats: list[float] = []

    # standardized numeric features
    for col in NUMERIC_COLUMNS:
        x = _safe_float(row[col])
        feats.append((x - prep.means[col]) / prep.stds[col])

    # engineered features
    old_org = _safe_float(row["oldbalanceOrg"])
    new_org = _safe_float(row["newbalanceOrig"])
    old_dest = _safe_float(row["oldbalanceDest"])
    new_dest = _safe_float(row["newbalanceDest"])
    amount = _safe_float(row["amount"])

    origin_delta = old_org - new_org
    dest_delta = new_dest - old_dest
    origin_error = origin_delta - amount
    dest_error = amount - dest_delta

    feats.extend(
        [
            origin_delta,
            dest_delta,
            origin_error,
            dest_error,
        ]
    )

    # one-hot transaction type
    type_vec = np.zeros(len(prep.type_to_idx), dtype=np.float32)
    t = str(row["type"]).strip()
    if t in prep.type_to_idx:
        type_vec[prep.type_to_idx[t]] = 1.0

    feats.extend(type_vec.tolist())
    return np.asarray(feats, dtype=np.float32)


def _hf_dataset_to_arrays(ds: HFDataset) -> tuple[np.ndarray, np.ndarray]:
    prep = _ensure_preprocessor()

    features = np.zeros((len(ds), prep.input_dim), dtype=np.float32)
    labels = np.zeros(len(ds), dtype=np.float32)

    for i, row in enumerate(ds):
        features[i] = _encode_row(row, prep)
        labels[i] = float(row[LABEL_COL])

    return features, labels


def _extract_partition_id(ds: HFDataset, fallback_partition_id: int | str) -> int | str:
    """Return the natural bank id when the partition contains exactly one."""
    if PARTITION_COL in ds.column_names and len(ds) > 0:
        bank_ids = {int(x) for x in ds[PARTITION_COL]}
        if len(bank_ids) == 1:
            return next(iter(bank_ids))
    return fallback_partition_id


def _dataset_fingerprint(ds: HFDataset) -> str:
    """Build a deterministic fingerprint for one fraud partition."""
    hasher = hashlib.sha256()
    stable_columns = [
        *NUMERIC_COLUMNS,
        "type",
        LABEL_COL,
        PARTITION_COL,
    ]
    available_columns = [col for col in stable_columns if col in ds.column_names]

    for idx, row in enumerate(ds):
        hasher.update(str(idx).encode("utf-8"))
        for col in available_columns:
            hasher.update(b"|")
            hasher.update(str(row[col]).encode("utf-8"))
        hasher.update(b"\n")

    return hasher.hexdigest()


def build_partition_metadata(
    ds: HFDataset,
    *,
    dataset_version: str,
    fallback_partition_id: int | str,
) -> dict[str, int | str]:
    """Return metadata used by benchmark preflight verification."""
    return {
        "partition_id": _extract_partition_id(ds, fallback_partition_id),
        "dataset_version": dataset_version,
        "dataset_fingerprint": _dataset_fingerprint(ds),
        "num_examples": len(ds),
    }


def _build_dataloader(
    ds: HFDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    features, labels = _hf_dataset_to_arrays(ds)
    torch_ds = FraudDataset(features, labels)

    if shuffle:
        labels_np = labels.astype(np.int64)
        class_counts = np.bincount(labels_np, minlength=2).astype(np.float64)
        class_weights = np.zeros_like(class_counts, dtype=np.float64)
        nonzero = class_counts > 0
        class_weights[nonzero] = 1.0 / class_counts[nonzero]

        sample_weights = class_weights[labels_np]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            torch_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=False,
        )

    return DataLoader(
        torch_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def load_sim_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partitioner_name: str = "iid",
) -> DataLoader:
    """
    Simulation mode:
      - use FederatedDataset on the HF train split
      - partition either IID or naturally by BankID
    """
    global fds

    normalized_partitioner = partitioner_name.strip().lower()

    if fds is None:
        partitioner = _make_partitioner(normalized_partitioner, num_partitions)
        fds = FederatedDataset(
            dataset=HF_DATASET_ID,
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id, "train")
    return _build_dataloader(partition, batch_size=batch_size, shuffle=True)


def load_sim_metadata(
    partition_id: int,
    num_partitions: int,
    partitioner_name: str = "iid",
) -> dict[str, int | str]:
    """Load simulation partition metadata without constructing a DataLoader."""
    global fds

    normalized_partitioner = partitioner_name.strip().lower()

    if fds is None:
        partitioner = _make_partitioner(normalized_partitioner, num_partitions)
        fds = FederatedDataset(
            dataset=HF_DATASET_ID,
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id, "train")
    return build_partition_metadata(
        partition,
        dataset_version=HF_DATASET_ID,
        fallback_partition_id=partition_id,
    )


def load_local_data(
    data_path: str,
    batch_size: int,
    split: str = "train",
) -> DataLoader:
    """
    Deployment mode:
      - if load_from_disk(data_path) is a DatasetDict, use the requested split
      - otherwise assume it is already the desired split
    """
    ds_or_dict = load_from_disk(data_path)

    if hasattr(ds_or_dict, "keys") and split in ds_or_dict:
        ds = ds_or_dict[split]
    else:
        ds = ds_or_dict

    return _build_dataloader(ds, batch_size=batch_size, shuffle=(split == "train"))


def load_local_metadata(
    data_path: str,
    *,
    dataset_version: str = "local",
    fallback_partition_id: int | str = "unknown",
    split: str = "train",
) -> dict[str, int | str]:
    """Load deployment partition metadata from a local dataset directory."""
    ds_or_dict = load_from_disk(data_path)

    if hasattr(ds_or_dict, "keys") and split in ds_or_dict:
        ds = ds_or_dict[split]
    else:
        ds = ds_or_dict

    return build_partition_metadata(
        ds,
        dataset_version=dataset_version,
        fallback_partition_id=fallback_partition_id,
    )


def load_centralized_dataset(batch_size: int = 1024) -> DataLoader:
    """Server-side centralized evaluation loader using the HF test split."""
    test_ds = load_dataset(HF_DATASET_ID, split="test")
    return _build_dataloader(test_ds, batch_size=batch_size, shuffle=False)


def _compute_pos_weight_from_loader(trainloader: DataLoader, device: torch.device) -> torch.Tensor:
    pos = 0.0
    total = 0.0

    for _, labels in trainloader:
        pos += float(labels.sum().item())
        total += float(labels.numel())

    neg = total - pos

    if pos <= 0:
        # no positive samples on this client
        return torch.tensor(1.0, dtype=torch.float32, device=device)

    pos_weight = neg / pos

    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    use_class_weights: bool = True,
    show_progress: bool = False,
):
    """Train one local client model."""
    net.to(device)

    if use_class_weights:
        pos_weight = _compute_pos_weight_from_loader(trainloader, device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    net.train()
    running_loss = 0.0
    num_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        progress_bar = tqdm(
            trainloader,
            desc=f"Train Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
            disable=not show_progress,
        )

        for features, labels in progress_bar:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = net(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value
            epoch_loss += loss_value
            num_steps += 1

            progress_bar.set_postfix(loss=f"{loss_value:.4f}")

        mean_epoch_loss = epoch_loss / max(len(trainloader), 1)
        tqdm.write(f"[train] epoch={epoch + 1}/{epochs} mean_loss={mean_epoch_loss:.4f}")

    return running_loss / max(num_steps, 1)


def _binary_classification_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.05,
) -> dict[str, float]:
    preds = (probs >= threshold).float()

    tp = float(((preds == 1) & (labels == 1)).sum().item())
    tn = float(((preds == 0) & (labels == 0)).sum().item())
    fp = float(((preds == 1) & (labels == 0)).sum().item())
    fn = float(((preds == 0) & (labels == 1)).sum().item())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # average_precision_score is the standard summary for the PR curve
    # and is commonly used as PR-AUC in imbalanced classification
    pr_auc = float(average_precision_score(labels_np, probs_np))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
    }

def _evaluate_thresholds(
    probs: torch.Tensor, 
    labels: torch.Tensor, 
    thresholds: list[float],
    ) -> dict[float, dict[str, float]]:
    """Evaluate metrics at multiple thresholds."""
    results = {}
    for t in thresholds:
        metrics = _binary_classification_metrics(probs, labels, threshold=t)
        results[t] = metrics
    return results


def test(net: nn.Module, testloader: DataLoader, device: torch.device):
    """Evaluate on the centralized HF test split."""
    net.to(device)
    net.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    num_batches = 0

    all_probs = []
    all_labels = []

    progress_bar = tqdm(
        testloader,
        desc="Validation",
        unit="batch",
        leave=False,
    )

    with torch.no_grad():
        for features, labels in progress_bar:
            features = features.to(device)
            labels = labels.to(device)

            logits = net(features)
            loss = loss_fn(logits, labels)

            probs = torch.sigmoid(logits)

            total_loss += float(loss.item())
            num_batches += 1

            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())

            progress_bar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    mean_loss = total_loss / max(num_batches, 1)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    thresholds = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
    threshold_metrics = _evaluate_thresholds(probs, labels, thresholds=thresholds)
    for t in thresholds:
        m = threshold_metrics[t]
        tqdm.write(
            f"[val] threshold={t:.2f} "
            f"loss={mean_loss:.4f} "
            f"acc={m['accuracy']:.4f} "
            f"precision={m['precision']:.4f} "
            f"recall={m['recall']:.4f} "
            f"f1={m['f1']:.4f} "
            f"pr_auc={m['pr_auc']:.6f}"
        )
    return mean_loss, threshold_metrics


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))
