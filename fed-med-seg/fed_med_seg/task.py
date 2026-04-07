"""fed_med_seg: BraTS federated segmentation task."""

from pathlib import Path
from typing import Any

import math
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, NaturalIdPartitioner
from huggingface_hub import snapshot_download
from monai.data import CacheDataset, DataLoader, decollate_batch, NibabelReader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)
from tqdm.auto import tqdm

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
HF_DATASET_ID = "flwrlabs/fed-brats"

fds = None
_hf_repo_root = None


class Net(UNet):
    """3D U-Net for BraTS segmentation."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )


def _ensure_local_hf_repo() -> Path:
    """Download/cache the full dataset repo and return its local path."""
    global _hf_repo_root

    if _hf_repo_root is None:
        repo_dir = snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
        )
        _hf_repo_root = Path(repo_dir).resolve()

    return _hf_repo_root


def _dataset_base_dir(ds: HFDataset) -> Path:
    """Resolve the base directory for repo-relative file paths."""
    cache_files = getattr(ds, "cache_files", None)
    if not cache_files:
        # Fallback to the full local HF repo snapshot
        return _ensure_local_hf_repo()

    first_file = Path(cache_files[0]["filename"]).resolve()

    # If loaded from local repo snapshot, parquet files live in <repo>/metadata/*.parquet
    if first_file.parent.name == "metadata":
        return first_file.parent.parent

    # Saved partitions via save_to_disk() store arrow files elsewhere; those rows still
    # contain repo-relative paths, so prefer the original downloaded repo when needed.
    local_repo = _ensure_local_hf_repo()
    if (local_repo / "data").exists():
        return local_repo

    return first_file.parent


def _resolve_repo_relative_path(base_dir: Path, rel_path: str) -> Path | None:
    """Resolve a repo-relative path against likely roots without dereferencing symlinks."""
    rel_path = str(rel_path)
    candidates = [
        base_dir / rel_path,
        _ensure_local_hf_repo() / rel_path,
    ]

    for path in candidates:
        if path.exists():
            return path.absolute()

    return None


def _row_to_case_item(row: dict[str, Any], base_dir: Path) -> dict[str, Any] | None:
    """Convert one HF dataset row to the MONAI dict expected by the transforms."""
    subject_id = str(row["BraTS Subject ID"]).strip()
    site = str(row["Site"]).strip()

    item = {
        "case_id": subject_id,
        "site": site,
    }

    for mod in MODALITIES:
        rel = row.get(mod)
        if rel is None:
            return None
        abs_path = _resolve_repo_relative_path(base_dir, str(rel))
        if abs_path is None:
            return None
        item[f"image_{mod}"] = str(abs_path)

    mask_rel = row.get("Mask")
    if mask_rel is None:
        return None
    mask_path = _resolve_repo_relative_path(base_dir, str(mask_rel))
    if mask_path is None:
        return None
    item["label"] = str(mask_path)

    return item


def _hf_dataset_to_case_items(ds: HFDataset) -> list[dict[str, Any]]:
    """Convert an HF dataset split/partition into MONAI case dicts."""
    base_dir = _dataset_base_dir(ds)
    cases: list[dict[str, Any]] = []
    skipped = 0

    for row in ds:
        item = _row_to_case_item(row, base_dir)
        if item is None:
            skipped += 1
            continue
        cases.append(item)

    if not cases:
        raise RuntimeError(
            "No usable cases found in the Hugging Face dataset split. "
            f"Resolved base_dir={base_dir}"
        )

    if skipped:
        print(f"Warning: skipped {skipped} rows with missing files")

    return cases


def _make_partitioner(partitioner_name: str, num_partitions: int):
    name = partitioner_name.strip().lower()

    if name == "iid":
        return IidPartitioner(num_partitions=num_partitions)

    if name == "natural":
        return NaturalIdPartitioner(partition_by="Site")

    raise ValueError(
        f"Unsupported partitioner '{partitioner_name}'. "
        "Use 'iid' or 'natural'."
    )


def get_train_transforms(roi_size: tuple[int, int, int]):
    image_keys = [f"image_{m}" for m in MODALITIES]

    return Compose(
        [
            LoadImaged(keys=image_keys + ["label"], reader=NibabelReader()),
            EnsureChannelFirstd(keys=image_keys + ["label"]),
            Spacingd(
                keys=image_keys + ["label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
            ),
            Orientationd(keys=image_keys + ["label"], axcodes="RAS"),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            MapLabelValued(
                keys=["label"],
                orig_labels=[0, 1, 2, 4],
                target_labels=[0, 1, 2, 3],
            ),
            ConcatItemsd(keys=image_keys, name="image", dim=0),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms():
    image_keys = [f"image_{m}" for m in MODALITIES]

    return Compose(
        [
            LoadImaged(keys=image_keys + ["label"], reader=NibabelReader()),
            EnsureChannelFirstd(keys=image_keys + ["label"]),
            Spacingd(
                keys=image_keys + ["label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
            ),
            Orientationd(keys=image_keys + ["label"], axcodes="RAS"),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            MapLabelValued(
                keys=["label"],
                orig_labels=[0, 1, 2, 4],
                target_labels=[0, 1, 2, 3],
            ),
            ConcatItemsd(keys=image_keys, name="image", dim=0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def _build_trainloader(
    train_cases: list[dict[str, Any]],
    batch_size: int,
    roi_size: tuple[int, int, int],
    cache_rate: float = 0.25,
) -> DataLoader:
    train_ds = CacheDataset(
        data=train_cases,
        transform=get_train_transforms(roi_size),
        cache_rate=cache_rate,
        num_workers=2,
    )
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )


def _build_valloader(
    val_cases: list[dict[str, Any]],
    batch_size: int,
    cache_rate: float = 0.25,
) -> DataLoader:
    val_ds = CacheDataset(
        data=val_cases,
        transform=get_val_transforms(),
        cache_rate=cache_rate,
        num_workers=2,
    )
    return DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )


def load_sim_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    roi_size: tuple[int, int, int],
    partitioner_name: str = "iid",
    cache_rate: float = 0.25,
) -> DataLoader:
    """
    Simulation mode:
      - use FederatedDataset on the HF dataset train split
      - partition that split across clients
      - convert each partition row into MONAI file-path samples

    Supported partitioner_name values:
      - 'iid'
      - 'natural'  (NaturalIdPartitioner(partition_by="Site"))
    """
    global fds

    normalized_partitioner = partitioner_name.strip().lower()
    repo_dir = _ensure_local_hf_repo()

    if fds is None:
        partitioner = _make_partitioner(normalized_partitioner, num_partitions)
        fds = FederatedDataset(
            dataset=str(repo_dir),
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    train_cases = _hf_dataset_to_case_items(partition)

    return _build_trainloader(
        train_cases=train_cases,
        batch_size=batch_size,
        roi_size=roi_size,
        cache_rate=cache_rate,
    )


def load_local_data(
    data_path: str,
    batch_size: int,
    roi_size: tuple[int, int, int],
    cache_rate: float = 0.25,
) -> DataLoader:
    """
    Deployment mode:
      - load pre-stored local data from load_from_disk(...)
      - saved partitions may still contain repo-relative paths, so resolution
        falls back to the downloaded HF repo snapshot.
    """
    train_ds = load_from_disk(data_path)
    train_cases = _hf_dataset_to_case_items(train_ds)

    return _build_trainloader(
        train_cases=train_cases,
        batch_size=batch_size,
        roi_size=roi_size,
        cache_rate=cache_rate,
    )


def load_centralized_dataset(
    cache_rate: float = 0.25,
    batch_size: int = 4,
) -> DataLoader:
    """Server-side centralized evaluation loader using the HF test split."""
    repo_dir = _ensure_local_hf_repo()
    test_ds = load_dataset(str(repo_dir), split="test")
    val_cases = _hf_dataset_to_case_items(test_ds)

    return _build_valloader(
        val_cases=val_cases,
        batch_size=batch_size,
        cache_rate=cache_rate,
    )


def train(net, trainloader, epochs, lr, device, show_progress = False):
    """Train one local client model."""
    net.to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
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

        for batch_data in progress_bar:
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device).long()

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value
            epoch_loss += loss_value
            num_steps += 1

            progress_bar.set_postfix(loss=f"{loss_value:.4f}")

        mean_epoch_loss = epoch_loss / max(len(trainloader), 1)
        tqdm.write(f"[train] epoch={epoch + 1}/{epochs} mean_loss={mean_epoch_loss:.4f}")

    return running_loss / max(num_steps, 1)


def test(net, testloader, device, num_classes: int = 4):
    """Evaluate on the server-side test split."""
    net.to(device)
    net.eval()

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    total_loss = 0.0
    num_batches = 0
    roi_size = (128, 128, 128)

    progress_bar = tqdm(
        testloader,
        desc="Validation",
        unit="batch",
        leave=False,
    )

    with torch.no_grad():
        for batch_data in progress_bar:
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device).long()

            outputs = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=net,
            )

            loss = loss_fn(outputs, labels)
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1

            outputs_list = [post_pred(o) for o in decollate_batch(outputs)]
            labels_list = [post_label(l) for l in decollate_batch(labels)]
            dice_metric(y_pred=outputs_list, y=labels_list)

            current_mean_loss = total_loss / num_batches
            progress_bar.set_postfix(loss=f"{current_mean_loss:.4f}")

    mean_loss = total_loss / max(num_batches, 1)
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    tqdm.write(f"[val] mean_loss={mean_loss:.4f} mean_dice={mean_dice:.4f}")
    return mean_loss, mean_dice


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))
