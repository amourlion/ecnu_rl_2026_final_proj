from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compute_candidate_recall(
    projects: pd.DataFrame,
    entries: pd.DataFrame,
    workers: pd.DataFrame,
    events: pd.DataFrame,
    split: str,
    candidate_k: int,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Measure whether the logged true project appears in the candidate set."""
    del entries
    split_events = (
        events.loc[events["split"] == split].copy()
        if "split" in events.columns
        else events.copy()
    )
    split_events = split_events.sort_values(["entry_created_at", "event_id"]).reset_index(drop=True)
    project_features = _ProjectFeatures.from_projects(projects)
    worker_categories: dict[int, list[int]] = {}

    total = 0
    natural_hits = 0
    final_hits = 0
    no_natural_candidates = 0
    no_final_candidates = 0

    for _, event in split_events.iterrows():
        if max_steps is not None and total >= max_steps:
            break
        true_project_id = int(event["project_id"])
        worker_id = int(event["worker_id"])
        preferred_category = _top_category(worker_categories.get(worker_id, []))
        candidate_ids = project_features.top_k_ids(
            current_time=pd.to_datetime(event["entry_created_at"], utc=True),
            preferred_category=preferred_category,
            candidate_k=candidate_k,
        )
        if len(candidate_ids) == 0:
            no_natural_candidates += 1
        elif true_project_id in candidate_ids:
            natural_hits += 1

        final_ids = _environment_style_candidate_ids(
            natural_ids=candidate_ids,
            project_ids=project_features.project_id_set,
            true_project_id=true_project_id,
            candidate_k=candidate_k,
        )
        if not final_ids:
            no_final_candidates += 1
        elif true_project_id in final_ids:
            final_hits += 1

        project_category = project_features.category_by_id.get(true_project_id)
        if project_category is not None:
            worker_categories.setdefault(worker_id, []).append(project_category)

        total += 1

    natural_misses = total - natural_hits
    final_misses = total - final_hits
    return {
        "split": split,
        "candidate_k": candidate_k,
        "events": total,
        "natural_hits": natural_hits,
        "natural_misses": natural_misses,
        "natural_recall_at_k": _safe_ratio(natural_hits, total),
        "natural_miss_rate_at_k": _safe_ratio(natural_misses, total),
        "final_hits": final_hits,
        "final_misses": final_misses,
        "final_recall_at_k": _safe_ratio(final_hits, total),
        "final_miss_rate_at_k": _safe_ratio(final_misses, total),
        "no_natural_candidates": no_natural_candidates,
        "no_final_candidates": no_final_candidates,
    }


def compute_candidate_recall_from_artifacts(
    artifact_dir: str | Path,
    candidate_k: int,
    splits: list[str],
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    artifact_dir = Path(artifact_dir)
    projects = pd.read_parquet(artifact_dir / "projects.parquet")
    entries = pd.read_parquet(artifact_dir / "entries.parquet")
    workers = pd.read_parquet(artifact_dir / "workers.parquet")
    events = pd.read_parquet(artifact_dir / "events.parquet")
    return [
        compute_candidate_recall(
            projects=projects,
            entries=entries,
            workers=workers,
            events=events,
            split=split,
            candidate_k=candidate_k,
            max_steps=max_steps,
        )
        for split in splits
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure candidate recall@K for the logged true project.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/processed"))
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    results = compute_candidate_recall_from_artifacts(
        artifact_dir=args.artifact_dir,
        candidate_k=args.candidate_k,
        splits=args.splits,
        max_steps=args.max_steps,
    )
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        _print_table(results)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


class _ProjectFeatures:
    def __init__(
        self,
        project_ids: np.ndarray,
        categories: np.ndarray,
        start_ns: np.ndarray,
        deadline_ns: np.ndarray,
        total_awards: np.ndarray,
        average_scores: np.ndarray,
        competitions: np.ndarray,
    ) -> None:
        self.project_ids = project_ids
        self.categories = categories
        self.start_ns = start_ns
        self.deadline_ns = deadline_ns
        self.total_awards = total_awards
        self.average_scores = average_scores
        self.competitions = competitions
        self.project_id_set = set(int(project_id) for project_id in project_ids)
        self.category_by_id = {
            int(project_id): int(category)
            for project_id, category in zip(project_ids, categories, strict=False)
        }

    @classmethod
    def from_projects(cls, projects: pd.DataFrame) -> "_ProjectFeatures":
        return cls(
            project_ids=projects["project_id"].astype(int).to_numpy(),
            categories=projects["category"].astype(int).to_numpy(),
            start_ns=_datetime_ns(projects["start_date"]),
            deadline_ns=_datetime_ns(projects["deadline"]),
            total_awards=projects["total_awards"].astype(float).to_numpy(),
            average_scores=projects["average_score"].astype(float).to_numpy(),
            competitions=projects["entry_count"].clip(lower=0).astype(float).to_numpy(),
        )

    def top_k_ids(
        self,
        current_time: pd.Timestamp,
        preferred_category: int | None,
        candidate_k: int,
    ) -> set[int]:
        current_ns = current_time.value
        active = (self.start_ns <= current_ns) & (self.deadline_ns >= current_ns)
        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            return set()

        category_match = (
            self.categories[active_idx] == preferred_category
            if preferred_category is not None
            else np.zeros(active_idx.size, dtype=bool)
        )
        remaining_hours = (self.deadline_ns[active_idx] - current_ns) / 3_600_000_000_000.0
        scores = (
            category_match.astype(float) * 2.0
            + _rank_pct(self.total_awards[active_idx])
            + _rank_pct(self.average_scores[active_idx])
            - _rank_pct(self.competitions[active_idx]) * 0.25
            - _rank_pct(remaining_hours) * 0.05
        )
        order = np.lexsort((-self.total_awards[active_idx], -scores))
        top_idx = active_idx[order[:candidate_k]]
        return set(int(project_id) for project_id in self.project_ids[top_idx])


def _rank_pct(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks / values.size


def _datetime_ns(values: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(values, utc=True)
        .dt.tz_convert(None)
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy()
    )


def _top_category(categories: list[int]) -> int | None:
    if not categories:
        return None
    return max(set(categories), key=categories.count)


def _environment_style_candidate_ids(
    natural_ids: set[int],
    project_ids: set[int],
    true_project_id: int,
    candidate_k: int,
) -> set[int]:
    if true_project_id in natural_ids or true_project_id not in project_ids:
        return natural_ids
    final_ids = set(natural_ids)
    if len(final_ids) >= candidate_k:
        final_ids.remove(next(iter(final_ids)))
    final_ids.add(true_project_id)
    return final_ids


def _print_table(results: list[dict[str, Any]]) -> None:
    headers = [
        "split",
        "events",
        "natural_recall@k",
        "natural_miss@k",
        "final_recall@k",
        "final_miss@k",
    ]
    print("\t".join(headers))
    for row in results:
        print(
            "\t".join(
                [
                    str(row["split"]),
                    str(row["events"]),
                    f"{row['natural_recall_at_k']:.6f}",
                    f"{row['natural_miss_rate_at_k']:.6f}",
                    f"{row['final_recall_at_k']:.6f}",
                    f"{row['final_miss_rate_at_k']:.6f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
