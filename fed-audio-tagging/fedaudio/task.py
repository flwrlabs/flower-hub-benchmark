"""fedaudio: A Flower / PyTorch app (federated audio tagging)."""

from dataclasses import dataclass
from typing import Tuple

import io
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from datasets import Audio, Dataset, load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, NaturalIdPartitioner
from torch.utils.data import DataLoader

HF_DATASET = "flwrlabs/fed-urbansound8K" 
NUM_CLASSES = 10
PARTITION_COL = "clientID"

fds = None  # Cache FederatedDataset


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for audio preprocessing."""

    target_sr: int = 16000
    clip_seconds: float = 4.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64

    @property
    def target_samples(self) -> int:
        return int(self.target_sr * self.clip_seconds)


_AUDIO_CFG = AudioConfig()


def _pad_or_trim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad (with zeros) or trim to a fixed number of samples."""
    if x.numel() == target_len:
        return x
    if x.numel() > target_len:
        return x[:target_len]
    pad = target_len - x.numel()
    return F.pad(x, (0, pad))


def _make_frontend(cfg: AudioConfig):
    """Create reusable torchaudio transforms."""
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.target_sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
    return mel, to_db


_MEL, _TO_DB = _make_frontend(_AUDIO_CFG)


def _audio_to_logmel(batch):
    """Convert raw audio to log-mel spectrogram features."""
    audio_col = batch["audio"]
    class_ids = batch["classID"]

    is_batched = isinstance(audio_col, list)
    if not is_batched:
        audio_col = [audio_col]
        class_ids = [class_ids]

    feats = []
    labels = []

    for audio, cid in zip(audio_col, class_ids):
        audio_bytes = audio.get("bytes", None)
        audio_path = audio.get("path", None)

        if audio_bytes is not None:
            y_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        elif audio_path is not None and os.path.exists(audio_path):
            y_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        else:
            raise FileNotFoundError(
                f"Could not access audio example. path={audio_path!r}, "
                f"bytes_present={audio_bytes is not None}"
            )

        y = torch.tensor(y_np, dtype=torch.float32)

        # Ensure mono
        if y.ndim > 1:
            # soundfile often returns [num_frames, channels]
            y = y.mean(dim=-1)

        sr = int(sr)

        if sr != _AUDIO_CFG.target_sr:
            y = torchaudio.functional.resample(
                y, orig_freq=sr, new_freq=_AUDIO_CFG.target_sr
            )

        y = _pad_or_trim(y, _AUDIO_CFG.target_samples)

        mel = _MEL(y)
        logmel = _TO_DB(mel)
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

        feats.append(logmel.unsqueeze(0))
        labels.append(int(cid))

    if is_batched:
        batch["features"] = feats
        batch["label"] = labels
        return batch

    batch["features"] = feats[0]
    batch["label"] = labels[0]
    return batch


class SmallAudioCNN(nn.Module):
    """A compact CNN for log-mel spectrogram classification."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # self.bn1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # self.bn2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # self.bn3 = nn.GroupNorm(8, 64)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.2)

        # Adaptive pooling makes input-time dimension flexible
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


def make_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    return SmallAudioCNN(num_classes=num_classes)


def collate_audio_batch(batch):
    """Collate function that only batches tensors/numerics we need.

    Avoids collating HF `audio` objects (AudioDecoder).
    """
    x = torch.stack([b["features"] for b in batch], dim=0)
    y = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"features": x, "label": y}


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


def _make_dataloader(dataset: Dataset, batch_size: int):
    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset = dataset.with_transform(lambda s: _audio_to_logmel(s))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_audio_batch,
    )


def load_sim_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partitioner_name: str = "iid",
) -> DataLoader:
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global fds
    normalized_partitioner = partitioner_name.strip().lower()
    if fds is None:
        partitioner = _make_partitioner(normalized_partitioner, num_partitions)
        fds = FederatedDataset(
            dataset=HF_DATASET,
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)

    return _make_dataloader(partition, batch_size)


def load_local_data(data_path: str, batch_size: int):
    """Load local data."""
    # Load dataset from disk
    local_data = load_from_disk(data_path)
    return _make_dataloader(local_data, batch_size)


def load_centralized_dataset(batch_size: int) -> DataLoader:
    test_dataset = load_dataset(HF_DATASET, split="test")
    test_dataset = test_dataset.cast_column("audio", Audio(decode=False))
    test_dataset = test_dataset.with_transform(lambda s: _audio_to_logmel(s))
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(model, trainloader, epochs, lr, device):
    """Train the model on the local dataset."""
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running = 0.0
    num_steps = 0
    for _ in range(epochs):
        for batch in trainloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            num_steps += 1

    return running / max(1, num_steps)


@torch.no_grad()
def test(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Validate the model on the test set."""
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        x = batch["features"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return total_loss / max(1, len(dataloader)), correct / max(1, total)


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))
