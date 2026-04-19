"""
Phishing URL Classification - Data Loading & Preprocessing

Simulation mode:
- use flwr-datasets + NaturalIdPartitioner on the Hugging Face dataset
  yangao381/fed-phishing-urls
- load only the client-local TRAIN partition

Deployment mode:
- load only local client TRAIN data from disk

Global server-side evaluation:
- load the single global TEST split from the same HF dataset
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, NaturalIdPartitioner
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler

SIM_FED_DATASET_ID = "flwrlabs/fed-phishing-urls"
PARTITION_COL = "client_id"

_FDS_CACHE: dict[str, FederatedDataset] = {}
_CENTRAL_TEST_CACHE: dict[str, HFDataset] = {}

PAD_IDX = 0
UNK_IDX = 1
VOCAB_SIZE = 258  # 256 bytes + PAD + UNK


def build_vocab() -> tuple[dict[int, int], dict[int, int | None]]:
    """Build byte-level vocabulary."""
    byte2idx = {i: i + 2 for i in range(256)}
    idx2byte = {i + 2: i for i in range(256)}
    idx2byte[PAD_IDX] = None
    idx2byte[UNK_IDX] = None
    return byte2idx, idx2byte


BYTE2IDX, _ = build_vocab()


def url_to_bytes(url: str) -> bytes:
    """Convert URL string to normalized bytes."""
    from urllib.parse import unquote_to_bytes

    try:
        return unquote_to_bytes(url.lower())
    except Exception:
        return url.lower().encode("utf-8", errors="replace")


def url_to_indices(url: str, byte2idx: dict[int, int], max_len: int = 256) -> list[int]:
    """Convert URL string to fixed-length list of byte token indices."""
    url_bytes = url_to_bytes(url)
    indices = [byte2idx.get(byte, UNK_IDX) for byte in url_bytes[:max_len]]
    while len(indices) < max_len:
        indices.append(PAD_IDX)
    return indices


class PhishingURLDataset(TorchDataset):
    """PyTorch Dataset for phishing URL classification."""

    def __init__(
        self,
        urls: list[str],
        labels: list[int],
        char2idx: dict[int, int],
        max_len: int = 256,
    ):
        self.urls = urls
        self.labels = labels
        self.char2idx = char2idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int):
        url = self.urls[idx]
        label = self.labels[idx]
        indices = url_to_indices(url, self.char2idx, self.max_len)

        return {
            "input_ids": torch.tensor(indices, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def create_train_dataloader(
    urls: list[str],
    labels: list[int],
    char2idx: dict[int, int],
    max_len: int = 256,
    batch_size: int = 128,
    num_workers: int = 2,
    use_weighted_sampler: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader for local client training."""
    dataset = PhishingURLDataset(urls, labels, char2idx, max_len)

    sampler = None
    shuffle = True
    if use_weighted_sampler and len(labels) > 0:
        class_counts = Counter(labels)
        weights = [1.0 / class_counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_eval_dataloader(
    urls: list[str],
    labels: list[int],
    char2idx: dict[int, int],
    max_len: int = 256,
    batch_size: int = 128,
    num_workers: int = 2,
) -> DataLoader:
    """Create a PyTorch DataLoader for centralized evaluation."""
    dataset = PhishingURLDataset(urls, labels, char2idx, max_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_class_weights(labels: list[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """Compute BCE positive-class weight for imbalanced training."""
    num_pos = int(sum(labels))
    num_neg = int(len(labels) - num_pos)

    if num_pos == 0:
        pos_weight = torch.tensor([1.0], dtype=torch.float32)
    else:
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32)

    if device is not None:
        pos_weight = pos_weight.to(device)
    return pos_weight


def _extract_columns(hf_ds: HFDataset) -> tuple[list[str], list[int]]:
    """Read URL/text and label columns from a HF dataset."""
    if "url" in hf_ds.column_names:
        urls = list(hf_ds["url"])
    elif "text" in hf_ds.column_names:
        urls = list(hf_ds["text"])
    else:
        raise ValueError(
            f"Dataset must contain 'url' or 'text'. Found columns: {hf_ds.column_names}"
        )

    if "label" not in hf_ds.column_names:
        raise ValueError(f"Dataset must contain 'label'. Found: {hf_ds.column_names}")

    labels = [int(x) for x in hf_ds["label"]]
    return urls, labels


def _make_trainloader_from_split(
    train_ds: HFDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, torch.Tensor]:
    """Build local train loader and pos_weight from a split."""
    train_urls, train_labels = _extract_columns(train_ds)
    pos_weight = get_class_weights(train_labels, device)
    trainloader = create_train_dataloader(
        train_urls,
        train_labels,
        BYTE2IDX,
        batch_size=batch_size,
    )
    return trainloader, pos_weight


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


def _load_sim_fds(
    dataset_id: str = SIM_FED_DATASET_ID, 
    partitioner_name: str = "iid", 
    num_partitions: int = 5,
) -> FederatedDataset:
    """Load the simulation dataset through flwr-datasets and cache it."""
    if dataset_id not in _FDS_CACHE:
        normalized_partitioner = partitioner_name.strip().lower()
        train_partitioner = _make_partitioner(normalized_partitioner, num_partitions)
        _FDS_CACHE[dataset_id] = FederatedDataset(
            dataset=dataset_id,
            partitioners={"train": train_partitioner},
        )
    return _FDS_CACHE[dataset_id]


def load_sim_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    device: torch.device,
    dataset_id: str = SIM_FED_DATASET_ID,
    partitioner_name: str = "iid",
) -> tuple[DataLoader, torch.Tensor]:
    """Load client-local TRAIN data for simulation mode."""
    fds = _load_sim_fds(dataset_id, partitioner_name, num_partitions)
    train_client_ds = fds.load_partition(partition_id, "train")

    if len(train_client_ds) == 0:
        raise ValueError(
            f"No training samples found for partition_id={partition_id} in {dataset_id}"
        )
    
    if partitioner_name.strip().lower() == "natural":
        client_ids = set(int(x) for x in train_client_ds[PARTITION_COL])
        if len(client_ids) != 1:
            raise RuntimeError("NaturalIdPartitioner returned a mixed-client partition.")

        natural_client_id = next(iter(client_ids))
        print(
            f"Loaded simulation partition {partition_id}/{num_partitions - 1} "
            f"(natural client_id={natural_client_id}) from {dataset_id}: "
            f"train={len(train_client_ds):,}"
        )

    return _make_trainloader_from_split(
        train_ds=train_client_ds,
        batch_size=batch_size,
        device=device,
    )


def load_local_data(
    data_path: str,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, torch.Tensor]:
    """Load local TRAIN data from disk for deployment mode.

    Supported formats:
    - DatasetDict with a 'train' split
    - single Dataset with columns 'url'/'text' and 'label'
    """
    ds = load_from_disk(data_path)

    if isinstance(ds, DatasetDict):
        if "train" not in ds:
            raise ValueError(
                f"Local DatasetDict at {data_path} must contain a 'train' split."
            )
        return _make_trainloader_from_split(
            train_ds=ds["train"],
            batch_size=batch_size,
            device=device,
        )

    if isinstance(ds, HFDataset):
        return _make_trainloader_from_split(
            train_ds=ds,
            batch_size=batch_size,
            device=device,
        )

    raise ValueError(
        f"Unsupported local dataset at {data_path}. Expected DatasetDict or Dataset."
    )


def load_centralized_dataset(
    batch_size: int,
    dataset_id: str = SIM_FED_DATASET_ID,
) -> DataLoader:
    """Load the single global test split for server-side centralized evaluation."""
    if dataset_id not in _CENTRAL_TEST_CACHE:
        ds = load_dataset(dataset_id)
        if "test" not in ds:
            raise ValueError(f"{dataset_id} must contain a 'test' split.")
        _CENTRAL_TEST_CACHE[dataset_id] = ds["test"]

    test_ds = _CENTRAL_TEST_CACHE[dataset_id]
    test_urls, test_labels = _extract_columns(test_ds)

    return create_eval_dataloader(
        test_urls,
        test_labels,
        BYTE2IDX,
        batch_size=batch_size,
    )
