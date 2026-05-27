from __future__ import annotations

import pandas as pd


def active_candidates(
    projects: pd.DataFrame,
    current_time: pd.Timestamp,
    worker_history: dict,
    candidate_k: int,
) -> pd.DataFrame:
    current_time = pd.to_datetime(current_time, utc=True)
    candidates = projects[
        (projects["start_date"] <= current_time) & (projects["deadline"] >= current_time)
    ].copy()
    if candidates.empty:
        return candidates

    preferred_category = worker_history.get("top_category")
    candidates["category_match"] = (
        candidates["category"].eq(preferred_category).astype(float)
        if preferred_category is not None
        else 0.0
    )
    remaining_hours = (
        (candidates["deadline"] - current_time).dt.total_seconds() / 3600.0
    ).clip(lower=0.0)
    candidates["remaining_hours"] = remaining_hours
    candidates["competition"] = candidates["entry_count"].clip(lower=0)
    candidates["is_active"] = True
    candidates["heuristic_score"] = (
        candidates["category_match"] * 2.0
        + candidates["total_awards"].rank(pct=True)
        + candidates["average_score"].rank(pct=True)
        - candidates["competition"].rank(pct=True) * 0.25
        - candidates["remaining_hours"].rank(pct=True) * 0.05
    )
    return (
        candidates.sort_values(["heuristic_score", "total_awards"], ascending=False)
        .head(candidate_k)
        .reset_index(drop=True)
    )

