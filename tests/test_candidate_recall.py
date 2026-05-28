from __future__ import annotations

import pandas as pd

from scripts.candidate_recall import compute_candidate_recall


def test_compute_candidate_recall_reports_natural_and_final_recall() -> None:
    projects = pd.DataFrame(
        {
            "project_id": [1, 2, 3],
            "category": [1, 1, 1],
            "sub_category": [1, 1, 1],
            "industry": [1, 1, 1],
            "start_date": pd.to_datetime(["2020-01-01"] * 3, utc=True),
            "deadline": pd.to_datetime(["2020-01-10"] * 3, utc=True),
            "total_awards": [300.0, 200.0, 100.0],
            "average_score": [5.0, 4.0, 3.0],
            "creative_count": [1, 1, 1],
            "entry_count": [1, 1, 1],
        }
    )
    workers = pd.DataFrame({"worker_id": [10], "worker_quality": [0.8]})
    events = pd.DataFrame(
        {
            "event_id": [0, 1],
            "worker_id": [10, 10],
            "project_id": [1, 3],
            "entry_created_at": pd.to_datetime(
                ["2020-01-02", "2020-01-03"],
                utc=True,
            ),
            "split": ["train", "train"],
            "score": [3.0, 3.0],
            "winner": [False, False],
            "finalist": [False, False],
            "withdrawn": [False, False],
            "award_value": [0.0, 0.0],
            "tip_value": [0.0, 0.0],
        }
    )

    result = compute_candidate_recall(
        projects=projects,
        entries=events,
        workers=workers,
        events=events,
        split="train",
        candidate_k=2,
    )

    assert result["events"] == 2
    assert result["natural_hits"] == 1
    assert result["natural_misses"] == 1
    assert result["natural_recall_at_k"] == 0.5
    assert result["final_hits"] == 2
    assert result["final_misses"] == 0
    assert result["final_recall_at_k"] == 1.0
