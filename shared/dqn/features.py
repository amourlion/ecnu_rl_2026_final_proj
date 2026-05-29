from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
        candidate_tensor = self._encode_candidates_batch(candidates)
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

    def _encode_candidates_batch(self, candidates: pd.DataFrame) -> torch.Tensor:
        """Encode all candidates at once — avoids per-row iterrows()."""
        if candidates.empty:
            return torch.empty((0, self.candidate_dim), dtype=torch.float32, device=self.device)

        def _num(col: str) -> np.ndarray:
            if col not in candidates.columns:
                return np.zeros(len(candidates), dtype=np.float64)
            series = pd.to_numeric(candidates[col], errors="coerce").fillna(0.0)
            return series.to_numpy(dtype=np.float64)

        rows = np.column_stack([
            np.log1p(np.maximum(_num("project_id"), 0.0)) / 12.0,
            np.log1p(np.maximum(_num("category"), 0.0)) / 12.0,
            np.log1p(np.maximum(_num("sub_category"), 0.0)) / 12.0,
            np.log1p(np.maximum(_num("industry"), 0.0)) / 12.0,
            np.log1p(np.maximum(_num("total_awards"), 0.0)) / 10.0,
            np.clip(_num("average_score"), 0.0, None) / 5.0,
            np.log1p(np.maximum(_num("creative_count"), 0.0)) / 12.0,
            np.log1p(np.maximum(_num("entry_count") if "entry_count" in candidates.columns else _num("competition"), 0.0)) / 12.0,
            np.nan_to_num(_num("category_match"), nan=0.0),
            np.clip(_num("remaining_hours"), 0.0, None) / (24.0 * 30.0),
            np.log1p(np.maximum(_num("competition"), 0.0)) / 12.0,
            np.nan_to_num(_num("heuristic_score"), nan=0.0),
        ])
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

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
