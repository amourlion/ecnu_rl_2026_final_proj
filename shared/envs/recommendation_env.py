from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.envs.candidates import active_candidates
from shared.envs.rewards import compute_rewards


class CrowdsourcingRecEnv:
    """Offline worker-arrival recommendation environment."""

    def __init__(
        self,
        projects: pd.DataFrame,
        entries: pd.DataFrame,
        workers: pd.DataFrame,
        events: pd.DataFrame,
        split: str = "train",
        candidate_k: int = 20,
        reward_type: str = "combined",
        alpha: float = 0.5,
    ) -> None:
        if reward_type not in {"worker", "requester", "combined"}:
            raise ValueError("reward_type must be worker, requester, or combined")
        self.projects = projects.copy()
        self.entries = entries.copy()
        self.workers = workers.copy()
        self.events = (
            events.loc[events["split"] == split].copy()
            if "split" in events.columns
            else events.copy()
        )
        self.events = self.events.sort_values(["entry_created_at", "event_id"]).reset_index(drop=True)
        self.split = split
        self.candidate_k = candidate_k
        self.reward_type = reward_type
        self.alpha = alpha
        self.index = 0
        self.worker_history: dict[int, dict] = {}
        self.worker_quality = self.workers.set_index("worker_id")["worker_quality"].to_dict()

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        split: str = "train",
        candidate_k: int = 20,
        reward_type: str = "combined",
        alpha: float = 0.5,
    ) -> "CrowdsourcingRecEnv":
        artifact_dir = Path(artifact_dir)
        return cls(
            pd.read_parquet(artifact_dir / "projects.parquet"),
            pd.read_parquet(artifact_dir / "entries.parquet"),
            pd.read_parquet(artifact_dir / "workers.parquet"),
            pd.read_parquet(artifact_dir / "events.parquet"),
            split=split,
            candidate_k=candidate_k,
            reward_type=reward_type,
            alpha=alpha,
        )

    def reset(self) -> dict:
        self.index = 0
        self.worker_history = {}
        return self._state()

    def _current_event(self) -> pd.Series:
        if self.index >= len(self.events):
            raise IndexError("environment is done")
        return self.events.iloc[self.index]

    def _state(self) -> dict:
        if self.index >= len(self.events):
            return {"done": True}
        event = self._current_event()
        return {
            "done": False,
            "event_id": int(event["event_id"]),
            "worker_id": int(event["worker_id"]),
            "current_time": event["entry_created_at"],
            "split": self.split,
        }

    def get_candidates(self) -> pd.DataFrame:
        event = self._current_event()
        worker_id = int(event["worker_id"])
        candidates = active_candidates(
            self.projects,
            event["entry_created_at"],
            self.worker_history.get(worker_id, {}),
            self.candidate_k,
        )
        true_project_id = int(event["project_id"])
        if true_project_id not in set(candidates.get("project_id", pd.Series(dtype=int)).astype(int)):
            true_project = self.projects[self.projects["project_id"] == true_project_id].copy()
            if not true_project.empty:
                current_time = pd.to_datetime(event["entry_created_at"], utc=True)
                true_project["category_match"] = 0.0
                true_project["remaining_hours"] = (
                    true_project["deadline"] - current_time
                ).dt.total_seconds() / 3600.0
                true_project["competition"] = true_project["entry_count"]
                true_project["is_active"] = (
                    (true_project["start_date"] <= current_time)
                    & (true_project["deadline"] >= current_time)
                )
                true_project["heuristic_score"] = -1.0
                candidates = pd.concat(
                    [
                        candidates.head(max(self.candidate_k - 1, 0)),
                        true_project,
                    ],
                    ignore_index=True,
                )
        return candidates.head(self.candidate_k).reset_index(drop=True)

    def step(self, action_index: int) -> tuple[dict, float, bool, dict]:
        candidates = self.get_candidates()
        if candidates.empty:
            raise ValueError("no candidates available for current event")
        if action_index < 0 or action_index >= len(candidates):
            raise IndexError("action_index must refer to a candidate row")

        event = self._current_event()
        recommended = candidates.iloc[action_index]
        true_project_id = int(event["project_id"])
        recommended_project_id = int(recommended["project_id"])
        hit = recommended_project_id == true_project_id
        worker_id = int(event["worker_id"])
        worker_quality = float(self.worker_quality.get(worker_id, 0.0))
        true_project = self.projects[self.projects["project_id"] == true_project_id]
        true_project_row = true_project.iloc[0] if not true_project.empty else None
        same_category = _same_value(recommended, true_project_row, "category")
        same_sub_category = _same_value(recommended, true_project_row, "sub_category")
        same_industry = _same_value(recommended, true_project_row, "industry")
        rewards = compute_rewards(
            hit=hit,
            score=float(event["score"]),
            winner=bool(event["winner"]),
            finalist=bool(event["finalist"]),
            withdrawn=bool(event["withdrawn"]),
            award_value=float(event["award_value"]),
            tip_value=float(event["tip_value"]),
            worker_quality=worker_quality,
            alpha=self.alpha,
            same_category=same_category,
            same_sub_category=same_sub_category,
            same_industry=same_industry,
        )

        info = {
            **rewards,
            "hit": hit,
            "recommended_project_id": recommended_project_id,
            "true_project_id": true_project_id,
            "worker_id": worker_id,
            "worker_quality": worker_quality,
            "score": float(event["score"]) if hit else 0.0,
            "winner": bool(event["winner"]) if hit else False,
            "finalist": bool(event["finalist"]) if hit else False,
            "withdrawn": bool(event["withdrawn"]) if hit else False,
            "category": int(recommended["category"]),
            "same_category": same_category,
            "same_sub_category": same_sub_category,
            "same_industry": same_industry,
        }
        self._update_history(worker_id, event)
        self.index += 1
        done = self.index >= len(self.events)
        reward = float(rewards[f"{self.reward_type}_reward"])
        return self._state(), reward, done, info

    def _update_history(self, worker_id: int, event: pd.Series) -> None:
        history = self.worker_history.setdefault(worker_id, {"categories": []})
        project = self.projects[self.projects["project_id"] == int(event["project_id"])]
        if not project.empty:
            history["categories"].append(int(project.iloc[0]["category"]))
            history["top_category"] = max(
                set(history["categories"]),
                key=history["categories"].count,
            )


def _same_value(left: pd.Series, right: pd.Series | None, column: str) -> bool:
    if right is None or column not in left or column not in right:
        return False
    return bool(left[column] == right[column])
