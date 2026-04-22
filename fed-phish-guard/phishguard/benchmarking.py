"""App-local benchmarking helpers for communication-cost tracking."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class RoundCommunication:
    """Per-round communication accounting."""

    bytes_down: int
    bytes_up: int
    num_messages_down: int
    num_messages_up: int

    @property
    def total_bytes(self) -> int:
        return self.bytes_down + self.bytes_up


def _recorddict_num_bytes(recorddict: RecordDict) -> int:
    num_bytes = 0
    for record in recorddict.values():
        count_bytes = getattr(record, "count_bytes", None)
        if callable(count_bytes):
            num_bytes += int(count_bytes())
    return num_bytes


def _message_num_bytes(message: Message) -> int:
    if message.has_content():
        return _recorddict_num_bytes(message.content)
    return 0


def _messages_num_bytes(messages: list[Message]) -> int:
    return sum(_message_num_bytes(message) for message in messages)


def _with_comm_metrics(
    metrics: MetricRecord | None, communication: RoundCommunication
) -> MetricRecord:
    metric_record = MetricRecord() if metrics is None else metrics
    metric_record["comm_bytes_down"] = communication.bytes_down
    metric_record["comm_bytes_up"] = communication.bytes_up
    metric_record["comm_bytes_total"] = communication.total_bytes
    metric_record["comm_num_messages_down"] = communication.num_messages_down
    metric_record["comm_num_messages_up"] = communication.num_messages_up
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
        result = Result()

        arrays = initial_arrays
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        for current_round in range(1, num_rounds + 1):
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
            )
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round, train_replies
            )
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            result.train_metrics_clientapp[current_round] = _with_comm_metrics(
                agg_train_metrics, train_comm
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
                )
                agg_evaluate_metrics = self.aggregate_evaluate(
                    current_round, evaluate_replies
                )
                result.evaluate_metrics_clientapp[current_round] = _with_comm_metrics(
                    agg_evaluate_metrics, evaluate_comm
                )

            if evaluate_fn:
                res = evaluate_fn(current_round, arrays)
                if res is not None:
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

    def _extract(metrics_by_round: dict[int, MetricRecord]) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for round_id, metrics in metrics_by_round.items():
            summary[str(round_id)] = {
                key: int(metrics.get(key, 0)) for key in COMM_KEYS
            }
        return summary

    return {
        "train_rounds": _extract(result.train_metrics_clientapp),
        "evaluate_rounds": _extract(result.evaluate_metrics_clientapp),
    }
