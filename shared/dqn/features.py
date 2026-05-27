from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass(slots=True)
class EncodedObservation:
    state: torch.Tensor
    candidates: torch.Tensor
    state_dim: int
    candidate_dim: int


class FeatureEncoder:
    """Numeric encoder for the shared crowdsourcing recommendation environment."""

    state_dim = 5
    candidate_dim = 12

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def encode(self, state: dict, candidates: pd.DataFrame) -> EncodedObservation:
        state_tensor = torch.tensor(
            [self._state_features(state)],
            dtype=torch.float32,
            device=self.device,
        )
        candidate_tensor = torch.tensor(
            [self._candidate_features(row) for _, row in candidates.iterrows()],
            dtype=torch.float32,
            device=self.device,
        )
        return EncodedObservation(
            state=state_tensor,
            candidates=candidate_tensor,
            state_dim=self.state_dim,
            candidate_dim=self.candidate_dim,
        )

    def _state_features(self, state: dict) -> list[float]:
        current_time = pd.to_datetime(state.get("current_time"), utc=True, errors="coerce")
        if pd.isna(current_time):
            hour = 0.0
            weekday = 0.0
        else:
            hour = float(current_time.hour) / 23.0
            weekday = float(current_time.dayofweek) / 6.0
        worker_id = float(state.get("worker_id", 0) or 0)
        event_id = float(state.get("event_id", 0) or 0)
        return [
            self._log_norm(worker_id),
            self._log_norm(event_id),
            hour,
            weekday,
            1.0 if state.get("split") == "train" else 0.0,
        ]

    def _candidate_features(self, row: pd.Series) -> list[float]:
        return [
            self._log_norm(row.get("project_id", 0.0)),
            self._log_norm(row.get("category", 0.0)),
            self._log_norm(row.get("sub_category", 0.0)),
            self._log_norm(row.get("industry", 0.0)),
            self._money_norm(row.get("total_awards", 0.0)),
            self._score_norm(row.get("average_score", 0.0)),
            self._log_norm(row.get("creative_count", 0.0)),
            self._log_norm(row.get("entry_count", row.get("competition", 0.0))),
            self._float(row.get("category_match", 0.0)),
            self._hours_norm(row.get("remaining_hours", 0.0)),
            self._log_norm(row.get("competition", 0.0)),
            self._float(row.get("heuristic_score", 0.0)),
        ]

    @staticmethod
    def _float(value: object) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        if pd.isna(result):
            return 0.0
        return result

    def _log_norm(self, value: object) -> float:
        value = max(self._float(value), 0.0)
        return float(torch.log1p(torch.tensor(value)).item() / 12.0)

    def _money_norm(self, value: object) -> float:
        value = max(self._float(value), 0.0)
        return float(torch.log1p(torch.tensor(value)).item() / 10.0)

    def _score_norm(self, value: object) -> float:
        return max(self._float(value), 0.0) / 5.0

    def _hours_norm(self, value: object) -> float:
        return max(self._float(value), 0.0) / (24.0 * 30.0)
