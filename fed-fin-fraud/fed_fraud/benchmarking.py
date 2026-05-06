"""App-local benchmarking helpers for communication and dataset verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

from flwr.app import Message, MessageType
from flwr.common.record.arrayrecord import ArrayRecord
from flwr.common.record.configrecord import ConfigRecord
from flwr.common.record.metricrecord import MetricRecord
from flwr.common.record.recorddict import RecordDict
from flwr.server.grid.grid import Grid
from flwr.serverapp.strategy import (
    FedAdagrad,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedProx,
    FedYogi,
)
from flwr.serverapp.strategy.result import Result


SUMMARY_KEYS = (
    "client_train_time_sec",
    "server_aggregation_time_sec",
    "client_peak_cpu_memory_mb",
    "client_peak_gpu_memory_mb",
    "round_wall_clock_sec",
)


@dataclass
class RoundCommunication:
    """Per-round communication accounting."""

    bytes_down: int
    bytes_up: int
    num_messages_down: int
    num_messages_up: int
    communication_time_sec: float

    @property
    def total_bytes(self) -> int:
        return self.bytes_down + self.bytes_up


def _recorddict_num_bytes(recorddict: RecordDict) -> int:
    num_bytes = 0
    for record in recorddict.values():
        num_bytes += int(record.count_bytes())
    return num_bytes


def _message_num_bytes(message: Message) -> int:
    if message.has_content():
        return _recorddict_num_bytes(message.content)
    return 0


def _messages_num_bytes(messages: list[Message]) -> int:
    return sum(_message_num_bytes(message) for message in messages)


def _with_comm_metrics(
    metrics: MetricRecord | None,
    communication: RoundCommunication,
    *,
    include_system_metrics: bool,
    aggregation_time_sec: float = 0.0,
    round_wall_clock_sec: float = 0.0,
) -> MetricRecord:
    metric_record = MetricRecord() if metrics is None else metrics
    metric_record["comm_bytes_total"] = communication.total_bytes
    if include_system_metrics:
        metric_record["server_aggregation_time_sec"] = float(aggregation_time_sec)
        metric_record["round_wall_clock_sec"] = float(round_wall_clock_sec)
    return metric_record


class CommunicationTrackingMixin:
    """Add benchmark behavior around an existing Flower strategy."""

    def __init__(
        self,
        *args,
        benchmark_verify_dataset: bool = False,
        benchmark_manifest_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benchmark_verify_dataset = benchmark_verify_dataset
        self.benchmark_manifest_path = benchmark_manifest_path
        self.verification_summary: dict[str, object] | None = None

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn=None,
    ) -> Result:
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        system_metrics_enabled = _system_metrics_enabled(train_config, evaluate_config)
        result = Result()

        if self.benchmark_verify_dataset:
            self.verification_summary = self._run_preflight_verification(grid, timeout)

        arrays = initial_arrays
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        for current_round in range(1, num_rounds + 1):
            round_start = time.perf_counter() if system_metrics_enabled else 0.0
            train_messages = list(
                self.configure_train(current_round, arrays, train_config, grid)
            )
            train_replies = list(
                grid.send_and_receive(messages=train_messages, timeout=timeout)
            )
            train_comm = RoundCommunication(
                bytes_down=_messages_num_bytes(train_messages),
                bytes_up=_messages_num_bytes(train_replies),
                num_messages_down=len(train_messages),
                num_messages_up=len(train_replies),
                communication_time_sec=0.0,
            )
            train_aggregation_start = (
                time.perf_counter() if system_metrics_enabled else 0.0
            )
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round, train_replies
            )
            train_aggregation_time_sec = (
                time.perf_counter() - train_aggregation_start
                if system_metrics_enabled
                else 0.0
            )
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            round_wall_clock_sec = (
                time.perf_counter() - round_start if system_metrics_enabled else 0.0
            )
            result.train_metrics_clientapp[current_round] = _with_comm_metrics(
                agg_train_metrics,
                train_comm,
                include_system_metrics=system_metrics_enabled,
                aggregation_time_sec=float(train_aggregation_time_sec),
                round_wall_clock_sec=float(round_wall_clock_sec),
            )

            evaluate_messages = list(
                self.configure_evaluate(current_round, arrays, evaluate_config, grid)
            )
            evaluate_replies = list(
                grid.send_and_receive(messages=evaluate_messages, timeout=timeout)
            )
            if evaluate_messages or evaluate_replies:
                evaluate_comm = RoundCommunication(
                    bytes_down=_messages_num_bytes(evaluate_messages),
                    bytes_up=_messages_num_bytes(evaluate_replies),
                    num_messages_down=len(evaluate_messages),
                    num_messages_up=len(evaluate_replies),
                    communication_time_sec=0.0,
                )
                agg_evaluate_metrics = self.aggregate_evaluate(
                    current_round, evaluate_replies
                )
                result.evaluate_metrics_clientapp[current_round] = _with_comm_metrics(
                    agg_evaluate_metrics,
                    evaluate_comm,
                    include_system_metrics=False,
                )

            if evaluate_fn:
                res = evaluate_fn(current_round, arrays)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        return result

    def _run_preflight_verification(self, grid: Grid, timeout: float) -> dict[str, object]:
        manifest = _load_manifest(self.benchmark_manifest_path)
        node_ids = list(grid.get_node_ids())
        verification_errors: list[str] = []

        expected_client_count = manifest.get("expected_client_count")
        if isinstance(expected_client_count, int) and expected_client_count != len(node_ids):
            verification_errors.append(
                f"expected {expected_client_count} connected clients, got {len(node_ids)}"
            )

        verify_config = ConfigRecord({"benchmark_mode": "verify_only"})
        verify_record = RecordDict({self.configrecord_key: verify_config})
        verify_messages = list(
            self._construct_messages(verify_record, node_ids, MessageType.TRAIN)
        )
        verify_replies = list(
            grid.send_and_receive(messages=verify_messages, timeout=timeout)
        )

        observed_clients: dict[str, dict[str, object]] = {}
        duplicate_partition_ids: set[str] = set()
        for reply in verify_replies:
            if not reply.has_content():
                verification_errors.append("client reply missing verification payload")
                continue
            metadata = reply.content.config_records.get("config")
            if metadata is None:
                verification_errors.append("client reply missing config record")
                continue

            partition_id = str(metadata.get("partition_id", "unknown"))
            client_summary = {
                "partition_id": partition_id,
                "dataset_version": str(metadata.get("dataset_version", "unknown")),
                "dataset_fingerprint": str(
                    metadata.get("dataset_fingerprint", "unknown")
                ),
                "num_examples": int(metadata.get("num_examples", 0)),
            }
            if partition_id in observed_clients:
                duplicate_partition_ids.add(partition_id)
            observed_clients[partition_id] = client_summary

        if duplicate_partition_ids:
            verification_errors.append(
                "duplicate partition ids reported: "
                + ", ".join(sorted(duplicate_partition_ids))
            )

        expected_clients = manifest.get("clients", {})
        if isinstance(expected_clients, dict):
            expected_partition_ids = sorted(str(key) for key in expected_clients.keys())
            missing_partition_ids = sorted(
                set(expected_partition_ids) - set(observed_clients.keys())
            )
            unexpected_partition_ids = sorted(
                set(observed_clients.keys()) - set(expected_partition_ids)
            )
            if missing_partition_ids:
                verification_errors.append(
                    "missing partitions: " + ", ".join(missing_partition_ids)
                )
            if unexpected_partition_ids:
                verification_errors.append(
                    "unexpected partitions: " + ", ".join(unexpected_partition_ids)
                )

            for partition_id, expected in expected_clients.items():
                if not isinstance(expected, dict):
                    continue
                observed = observed_clients.get(str(partition_id))
                if observed is None:
                    continue

                expected_dataset_version = expected.get("dataset_version")
                if (
                    expected_dataset_version
                    and str(expected_dataset_version) != observed["dataset_version"]
                ):
                    verification_errors.append(
                        f"dataset version mismatch for partition {partition_id}"
                    )

                expected_num_examples = expected.get("num_examples")
                if (
                    expected_num_examples is not None
                    and int(expected_num_examples) != observed["num_examples"]
                ):
                    verification_errors.append(
                        f"num_examples mismatch for partition {partition_id}"
                    )

                expected_fingerprint = expected.get("dataset_fingerprint")
                if (
                    expected_fingerprint
                    and str(expected_fingerprint) != observed["dataset_fingerprint"]
                ):
                    verification_errors.append(
                        f"fingerprint mismatch for partition {partition_id}"
                    )

        manifest_path = _manifest_path(self.benchmark_manifest_path)
        verification_summary = {
            "enabled": True,
            "manifest_path": str(manifest_path),
            "manifest_benchmark_id": manifest.get("benchmark_id", "unknown"),
            "passed": len(verification_errors) == 0,
            "connected_client_count": len(node_ids),
            "verified_reply_count": len(observed_clients),
            "observed_clients": observed_clients,
            "errors": verification_errors,
        }

        if verification_errors:
            raise RuntimeError(
                "Dataset fingerprint verification failed: "
                + "; ".join(verification_errors)
            )

        return verification_summary


class BenchmarkFedAvg(CommunicationTrackingMixin, FedAvg):
    """FedAvg with benchmark tracking."""


class BenchmarkFedProx(CommunicationTrackingMixin, FedProx):
    """FedProx with benchmark tracking."""


class BenchmarkFedAvgM(CommunicationTrackingMixin, FedAvgM):
    """FedAvgM with benchmark tracking."""


class BenchmarkFedAdam(CommunicationTrackingMixin, FedAdam):
    """FedAdam with benchmark tracking."""


class BenchmarkFedAdagrad(CommunicationTrackingMixin, FedAdagrad):
    """FedAdagrad with benchmark tracking."""


class BenchmarkFedYogi(CommunicationTrackingMixin, FedYogi):
    """FedYogi with benchmark tracking."""


def build_communication_summary(
    result: Result, verification_summary: dict[str, object] | None = None
) -> dict:
    """Create a compact JSON-serializable benchmark summary."""

    def _extract(metrics_by_round: dict[int, MetricRecord]) -> dict[str, dict[str, int | float]]:
        summary: dict[str, dict[str, int | float]] = {}
        for round_id, metrics in metrics_by_round.items():
            round_summary: dict[str, int | float] = {
                "comm_bytes_total": int(metrics.get("comm_bytes_total", 0))
            }
            for key in SUMMARY_KEYS:
                if key in metrics:
                    round_summary[key] = float(metrics.get(key, 0.0))
            summary[str(round_id)] = round_summary
        return summary

    train_rounds = _extract(result.train_metrics_clientapp)
    total_comm_bytes = sum(
        int(metrics.get("comm_bytes_total", 0))
        for metrics in result.train_metrics_clientapp.values()
    ) + sum(
        int(metrics.get("comm_bytes_total", 0))
        for metrics in result.evaluate_metrics_clientapp.values()
    )

    artifact = {
        "totals": {
            "total_comm_bytes": total_comm_bytes,
        },
        "train_rounds": train_rounds,
    }
    if verification_summary is not None:
        artifact["verification"] = verification_summary
    return artifact


def _system_metrics_enabled(
    train_config: ConfigRecord | None, evaluate_config: ConfigRecord | None
) -> bool:
    for config in (train_config, evaluate_config):
        if config is None:
            continue
        raw_value = config.get("benchmark-system-metrics", False)
        if str(raw_value).strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _manifest_path(manifest_path: str | None = None) -> Path:
    if manifest_path:
        return Path(manifest_path)
    return Path(__file__).with_name("benchmark_manifest.json")


def _load_manifest(manifest_path: str | None = None) -> dict[str, object]:
    with _manifest_path(manifest_path).open("r", encoding="utf-8") as f:
        return json.load(f)
