"""Export one deployment partition for fed-med-seg."""

from __future__ import annotations

import argparse
from pathlib import Path

from flwr_datasets import FederatedDataset

from fed_med_seg.task import _ensure_local_hf_repo, _make_partitioner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one fed-med-seg partition with the same partitioner "
            "used by simulation."
        )
    )
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--num-partitions", type=int, default=5)
    parser.add_argument("--partitioner", choices=("natural", "iid"), default="natural")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/fed-med-seg"),
        help="Base output directory. The script writes client_<partition-id>/ under it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = _ensure_local_hf_repo()
    partitioner = _make_partitioner(args.partitioner, args.num_partitions)
    fds = FederatedDataset(
        dataset=str(repo_dir),
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(args.partition_id)

    client_dir = args.output_dir / f"client_{args.partition_id}"
    client_dir.parent.mkdir(parents=True, exist_ok=True)
    partition.save_to_disk(str(client_dir))

    print(f"Saved partition to {client_dir}")
    print(f"Rows: {len(partition)}")


if __name__ == "__main__":
    main()
