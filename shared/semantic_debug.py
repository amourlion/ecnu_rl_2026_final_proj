from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from shared.envs.recommendation_env import CrowdsourcingRecEnv
from shared.metrics.evaluator import METRIC_FIELDS, evaluate_agent


def _fail(message: str) -> None:
    raise AssertionError(message)


def debug_processed_artifacts(artifact_dir: str | Path) -> dict:
    artifact_dir = Path(artifact_dir)
    required = {
        "projects": artifact_dir / "projects.parquet",
        "entries": artifact_dir / "entries.parquet",
        "workers": artifact_dir / "workers.parquet",
        "events": artifact_dir / "events.parquet",
        "splits": artifact_dir / "splits.json",
    }
    for name, path in required.items():
        if not path.exists():
            _fail(f"missing {name} artifact: {path}")

    projects = pd.read_parquet(required["projects"])
    entries = pd.read_parquet(required["entries"])
    workers = pd.read_parquet(required["workers"])
    events = pd.read_parquet(required["events"])
    splits = json.loads(required["splits"].read_text(encoding="utf-8"))

    if projects.empty:
        _fail("projects artifact is empty")
    if entries.empty:
        _fail("entries artifact is empty")
    if events.empty:
        _fail("events artifact is empty")
    if not events["entry_created_at"].is_monotonic_increasing:
        _fail("events are not sorted by entry_created_at")
    if not set(entries["project_id"]).issubset(set(projects["project_id"])):
        _fail("entries contain projects outside project artifact")
    if workers["worker_quality"].isna().any():
        _fail("worker_quality contains NaN after fill")
    if (workers["worker_quality"] < 0).any():
        _fail("worker_quality contains negative values after fill")
    split_total = sum(len(splits.get(name, [])) for name in ("train", "valid", "test"))
    if split_total != len(events):
        _fail("split id count does not match event count")

    return {
        "ok": True,
        "projects": int(len(projects)),
        "entries": int(len(entries)),
        "workers": int(len(workers)),
        "events": int(len(events)),
        "splits": {name: len(splits.get(name, [])) for name in ("train", "valid", "test")},
    }


def debug_environment_interface(
    artifact_dir: str | Path,
    split: str = "train",
    candidate_k: int = 20,
    steps: int = 100,
) -> dict:
    env = CrowdsourcingRecEnv.from_artifacts(
        artifact_dir,
        split=split,
        candidate_k=candidate_k,
        reward_type="combined",
    )
    state = env.reset()
    checked = 0
    done = bool(state.get("done", False))
    while not done and checked < steps:
        candidates = env.get_candidates()
        if candidates.empty:
            _fail("candidate set is empty")
        if len(candidates) > candidate_k:
            _fail("candidate set exceeds candidate_k")
        if not candidates["is_active"].all():
            _fail("candidate set contains inactive projects")
        next_state, reward, done, info = env.step(0)
        if not math.isfinite(float(reward)):
            _fail("reward is not finite")
        if info["recommended_project_id"] not in set(candidates["project_id"].astype(int)):
            _fail("recommended project is not from candidate set")
        state = next_state
        checked += 1

    return {"ok": True, "checked_steps": checked}


def debug_metrics_interface(
    artifact_dir: str | Path,
    split: str = "train",
    candidate_k: int = 20,
    steps: int | None = 100,
) -> dict:
    env = CrowdsourcingRecEnv.from_artifacts(artifact_dir, split=split, candidate_k=candidate_k)
    metrics = evaluate_agent("heuristic", env, max_steps=steps)
    missing = set(METRIC_FIELDS) - set(metrics)
    if missing:
        _fail(f"metrics missing fields: {sorted(missing)}")
    for key, value in metrics.items():
        if not math.isfinite(float(value)):
            _fail(f"metric {key} is not finite")
    return {"ok": True, "checked_steps": steps, "metrics": metrics}
