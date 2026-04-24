"""App-local benchmarking helpers for communication-cost tracking."""

from __future__ import annotations

from dataclasses import dataclass
import time

from flwr.app import Message
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


COMM_KEYS = (
    "comm_bytes_down",
    "comm_bytes_up",
    "comm_bytes_total",
    "comm_num_messages_down",
    "comm_num_messages_up",
)

SYSTEM_TIMING_KEYS = (
    "comm_time_sec",
    "server_aggregation_time_sec",
    "server_eval_time_sec",
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
    server_eval_time_sec: float = 0.0,
    round_wall_clock_sec: float = 0.0,
) -> MetricRecord:
    metric_record = MetricRecord() if metrics is None else metrics
    metric_record["comm_bytes_down"] = communication.bytes_down
    metric_record["comm_bytes_up"] = communication.bytes_up
    metric_record["comm_bytes_total"] = communication.total_bytes
    metric_record["comm_num_messages_down"] = communication.num_messages_down
    metric_record["comm_num_messages_up"] = communication.num_messages_up
    if include_system_metrics:
        metric_record["comm_time_sec"] = float(communication.communication_time_sec)
        metric_record["server_aggregation_time_sec"] = float(aggregation_time_sec)
        metric_record["server_eval_time_sec"] = float(server_eval_time_sec)
        metric_record["round_wall_clock_sec"] = float(round_wall_clock_sec)
    return metric_record


class CommunicationTrackingMixin:
    """Add communication-cost tracking around an existing Flower strategy."""

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

        arrays = initial_arrays
        if evaluate_fn:
            server_eval_start = time.perf_counter() if system_metrics_enabled else 0.0
            res = evaluate_fn(0, initial_arrays)
            server_eval_time_sec = (
                time.perf_counter() - server_eval_start if system_metrics_enabled else 0.0
            )
            if res is not None:
                if system_metrics_enabled:
                    res["server_eval_time_sec"] = float(server_eval_time_sec)
                result.evaluate_metrics_serverapp[0] = res

        for current_round in range(1, num_rounds + 1):
            round_start = time.perf_counter() if system_metrics_enabled else 0.0
            train_messages = list(
                self.configure_train(current_round, arrays, train_config, grid)
            )
            train_comm_start = time.perf_counter() if system_metrics_enabled else 0.0
            train_replies = list(
                grid.send_and_receive(messages=train_messages, timeout=timeout)
            )
            train_comm_time_sec = (
                time.perf_counter() - train_comm_start if system_metrics_enabled else 0.0
            )
            train_comm = RoundCommunication(
                bytes_down=_messages_num_bytes(train_messages),
                bytes_up=_messages_num_bytes(train_replies),
                num_messages_down=len(train_messages),
                num_messages_up=len(train_replies),
                communication_time_sec=float(train_comm_time_sec),
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
            round_elapsed_before_eval = (
                time.perf_counter() - round_start if system_metrics_enabled else 0.0
            )
            result.train_metrics_clientapp[current_round] = _with_comm_metrics(
                agg_train_metrics,
                train_comm,
                include_system_metrics=system_metrics_enabled,
                aggregation_time_sec=float(train_aggregation_time_sec),
                round_wall_clock_sec=float(round_elapsed_before_eval),
            )

            evaluate_messages = list(
                self.configure_evaluate(current_round, arrays, evaluate_config, grid)
            )
            evaluate_comm_start = (
                time.perf_counter() if system_metrics_enabled else 0.0
            )
            evaluate_replies = list(
                grid.send_and_receive(messages=evaluate_messages, timeout=timeout)
            )
            evaluate_comm_time_sec = (
                time.perf_counter() - evaluate_comm_start
                if system_metrics_enabled
                else 0.0
            )
            if evaluate_messages or evaluate_replies:
                evaluate_comm = RoundCommunication(
                    bytes_down=_messages_num_bytes(evaluate_messages),
                    bytes_up=_messages_num_bytes(evaluate_replies),
                    num_messages_down=len(evaluate_messages),
                    num_messages_up=len(evaluate_replies),
                    communication_time_sec=float(evaluate_comm_time_sec),
                )
                evaluate_aggregation_start = (
                    time.perf_counter() if system_metrics_enabled else 0.0
                )
                agg_evaluate_metrics = self.aggregate_evaluate(
                    current_round, evaluate_replies
                )
                evaluate_aggregation_time_sec = (
                    time.perf_counter() - evaluate_aggregation_start
                    if system_metrics_enabled
                    else 0.0
                )
                result.evaluate_metrics_clientapp[current_round] = _with_comm_metrics(
                    agg_evaluate_metrics,
                    evaluate_comm,
                    include_system_metrics=system_metrics_enabled,
                    aggregation_time_sec=float(evaluate_aggregation_time_sec),
                    round_wall_clock_sec=float(
                        time.perf_counter() - round_start
                        if system_metrics_enabled
                        else 0.0
                    ),
                )

            if evaluate_fn:
                server_eval_start = (
                    time.perf_counter() if system_metrics_enabled else 0.0
                )
                res = evaluate_fn(current_round, arrays)
                server_eval_time_sec = (
                    time.perf_counter() - server_eval_start
                    if system_metrics_enabled
                    else 0.0
                )
                if res is not None:
                    if system_metrics_enabled:
                        res["server_eval_time_sec"] = float(server_eval_time_sec)
                        res["round_wall_clock_sec"] = float(
                            time.perf_counter() - round_start
                        )
                    result.evaluate_metrics_serverapp[current_round] = res

        return result


class BenchmarkFedAvg(CommunicationTrackingMixin, FedAvg):
    """FedAvg with communication-cost tracking."""


class BenchmarkFedProx(CommunicationTrackingMixin, FedProx):
    """FedProx with communication-cost tracking."""


class BenchmarkFedAvgM(CommunicationTrackingMixin, FedAvgM):
    """FedAvgM with communication-cost tracking."""


class BenchmarkFedAdam(CommunicationTrackingMixin, FedAdam):
    """FedAdam with communication-cost tracking."""


class BenchmarkFedAdagrad(CommunicationTrackingMixin, FedAdagrad):
    """FedAdagrad with communication-cost tracking."""


class BenchmarkFedYogi(CommunicationTrackingMixin, FedYogi):
    """FedYogi with communication-cost tracking."""


def build_communication_summary(result: Result) -> dict:
    """Create a compact JSON-serializable communication summary."""

    def _extract(metrics_by_round: dict[int, MetricRecord]) -> dict[str, dict[str, int | float]]:
        summary: dict[str, dict[str, int | float]] = {}
        for round_id, metrics in metrics_by_round.items():
            round_summary: dict[str, int | float] = {
                key: int(metrics.get(key, 0)) for key in COMM_KEYS
            }
            for key in SYSTEM_TIMING_KEYS:
                if key in metrics:
                    round_summary[key] = float(metrics.get(key, 0.0))
            summary[str(round_id)] = round_summary
        return summary

    train_rounds = _extract(result.train_metrics_clientapp)
    evaluate_rounds = _extract(result.evaluate_metrics_clientapp)
    server_evaluate_rounds = {}
    for round_id, metrics in result.evaluate_metrics_serverapp.items():
        server_evaluate_rounds[str(round_id)] = {
            "server_eval_time_sec": float(metrics.get("server_eval_time_sec", 0.0)),
            "round_wall_clock_sec": float(metrics.get("round_wall_clock_sec", 0.0)),
        }

    total_comm_bytes_down = sum(
        int(metrics.get("comm_bytes_down", 0))
        for metrics in result.train_metrics_clientapp.values()
    ) + sum(
        int(metrics.get("comm_bytes_down", 0))
        for metrics in result.evaluate_metrics_clientapp.values()
    )
    total_comm_bytes_up = sum(
        int(metrics.get("comm_bytes_up", 0))
        for metrics in result.train_metrics_clientapp.values()
    ) + sum(
        int(metrics.get("comm_bytes_up", 0))
        for metrics in result.evaluate_metrics_clientapp.values()
    )
    total_comm_time_sec = sum(
        float(metrics.get("comm_time_sec", 0.0))
        for metrics in result.train_metrics_clientapp.values()
    ) + sum(
        float(metrics.get("comm_time_sec", 0.0))
        for metrics in result.evaluate_metrics_clientapp.values()
    )

    artifact = {
        "totals": {
            "total_comm_bytes_down": total_comm_bytes_down,
            "total_comm_bytes_up": total_comm_bytes_up,
            "total_comm_bytes": total_comm_bytes_down + total_comm_bytes_up,
        },
        "train_rounds": train_rounds,
        "evaluate_rounds": evaluate_rounds,
        "server_evaluate_rounds": server_evaluate_rounds,
    }
    if total_comm_time_sec > 0.0:
        artifact["totals"]["total_comm_time_sec"] = total_comm_time_sec
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
