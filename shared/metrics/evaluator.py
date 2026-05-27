from __future__ import annotations

from collections.abc import Callable

import pandas as pd


METRIC_FIELDS = [
    "avg_worker_reward",
    "avg_requester_reward",
    "avg_combined_reward",
    "hitrate_at_1",
    "avg_score",
    "winner_rate",
    "finalist_rate",
    "withdrawn_rate",
    "avg_worker_quality",
    "project_coverage",
    "category_diversity",
]


def _choose_action(agent, state: dict, candidates: pd.DataFrame) -> int:
    if candidates.empty:
        raise ValueError("cannot evaluate without candidates")
    if agent == "heuristic":
        return int(candidates["heuristic_score"].idxmax())
    if agent == "random":
        return 0
    if callable(agent):
        return int(agent(state, candidates))
    if hasattr(agent, "act"):
        return int(agent.act(state, candidates))
    raise ValueError("agent must be 'heuristic', 'random', callable, or expose act()")


def evaluate_agent(agent, env, max_steps: int | None = None) -> dict[str, float]:
    state = env.reset()
    rows = []
    done = bool(state.get("done", False))
    while not done:
        if max_steps is not None and len(rows) >= max_steps:
            break
        candidates = env.get_candidates()
        action = _choose_action(agent, state, candidates)
        state, _reward, done, info = env.step(action)
        rows.append(info)

    if not rows:
        return {field: 0.0 for field in METRIC_FIELDS}

    frame = pd.DataFrame(rows)
    recommended = frame["recommended_project_id"].nunique()
    total_projects = max(env.projects["project_id"].nunique(), 1)
    metrics = {
        "avg_worker_reward": float(frame["worker_reward"].mean()),
        "avg_requester_reward": float(frame["requester_reward"].mean()),
        "avg_combined_reward": float(frame["combined_reward"].mean()),
        "hitrate_at_1": float(frame["hit"].mean()),
        "avg_score": float(frame["score"].mean()),
        "winner_rate": float(frame["winner"].mean()),
        "finalist_rate": float(frame["finalist"].mean()),
        "withdrawn_rate": float(frame["withdrawn"].mean()),
        "avg_worker_quality": float(frame["worker_quality"].mean()),
        "project_coverage": float(recommended / total_projects),
        "category_diversity": float(frame["category"].nunique()),
    }
    return metrics
