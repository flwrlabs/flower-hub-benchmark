"""Export one deployment partition and its fingerprint metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flwr_datasets import FederatedDataset

from fed_fraud.task import (
    HF_DATASET_ID,
    build_partition_metadata,
    _make_partitioner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one fed-fin-fraud partition with the same partitioner "
            "used by simulation."
        )
    )
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--num-partitions", type=int, default=5)
    parser.add_argument("--partitioner", choices=("natural", "iid"), default="natural")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/fed-fin-fraud"),
        help="Base output directory. The script writes client_<partition-id>/ under it.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional metadata JSON path. Defaults to client_<partition-id>_metadata.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partitioner = _make_partitioner(args.partitioner, args.num_partitions)
    fds = FederatedDataset(
        dataset=HF_DATASET_ID,
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(args.partition_id, "train")

    client_dir = args.output_dir / f"client_{args.partition_id}"
    client_dir.parent.mkdir(parents=True, exist_ok=True)
    partition.save_to_disk(str(client_dir))

    metadata = build_partition_metadata(
        partition,
        dataset_version=HF_DATASET_ID,
        fallback_partition_id=args.partition_id,
    )
    metadata_path = (
        args.metadata_path
        if args.metadata_path is not None
        else args.output_dir / f"client_{args.partition_id}_metadata.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"Saved partition to {client_dir}")
    print(f"Saved metadata to {metadata_path}")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
