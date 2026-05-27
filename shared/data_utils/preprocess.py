from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from shared.data_utils.split import temporal_split, train_worker_ids, write_splits


@dataclass(frozen=True)
class ProcessedPaths:
    projects_path: Path
    entries_path: Path
    workers_path: Path
    events_path: Path
    splits_path: Path


PROJECT_COLUMNS = [
    "project_id",
    "listed_entry_count",
    "category",
    "sub_category",
    "industry",
    "start_date",
    "deadline",
    "entry_count",
    "average_score",
    "total_awards",
    "creative_count",
    "client_feedback",
]


ENTRY_COLUMNS = [
    "entry_id",
    "project_id",
    "entry_number",
    "worker_id",
    "entry_created_at",
    "winner",
    "finalist",
    "withdrawn",
    "eliminated",
    "award_value",
    "tip_value",
    "score",
]


def _read_project_list(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append((int(row[0]), int(row[1])))
    return rows


def _to_utc(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def _float_or_zero(value: object) -> float:
    if value is None or value == "":
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result):
        return 0.0
    return result


def _load_projects(data_dir: Path, project_list: list[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for project_id, listed_count in project_list:
        path = data_dir / "project" / f"project_{project_id}.txt"
        with path.open(encoding="utf-8") as handle:
            item = json.load(handle)
        rows.append(
            {
                "project_id": int(project_id),
                "listed_entry_count": int(listed_count),
                "category": int(item.get("category") or -1),
                "sub_category": int(item.get("sub_category") or -1),
                "industry": item.get("industry") or "unknown",
                "start_date": _to_utc(item.get("start_date")),
                "deadline": _to_utc(item.get("deadline")),
                "entry_count": int(item.get("entry_count") or 0),
                "average_score": _float_or_zero(item.get("average_score")),
                "total_awards": _float_or_zero(item.get("total_awards")),
                "creative_count": int(item.get("creative_count") or 0),
                "client_feedback": _float_or_zero(item.get("client_feedback")),
            }
        )
    projects = pd.DataFrame(rows, columns=PROJECT_COLUMNS)
    return projects[projects["deadline"] >= projects["start_date"]].reset_index(drop=True)


def _entry_score(item: dict) -> float:
    revisions = item.get("revisions") or []
    if not revisions:
        return 0.0
    return max(_float_or_zero(revision.get("score")) for revision in revisions)


def _load_entries(data_dir: Path, project_ids: set[int]) -> pd.DataFrame:
    rows = []
    entry_files: dict[int, list[Path]] = {project_id: [] for project_id in project_ids}
    pattern = re.compile(r"entry_(\d+)_\d+\.txt$")
    for path in (data_dir / "entry").glob("entry_*.txt"):
        match = pattern.match(path.name)
        if not match:
            continue
        project_id = int(match.group(1))
        if project_id in entry_files:
            entry_files[project_id].append(path)

    for project_id in sorted(project_ids):
        for path in sorted(entry_files.get(project_id, [])):
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            for item in payload.get("results") or []:
                if int(item.get("project")) not in project_ids:
                    continue
                rows.append(
                    {
                        "entry_id": int(item.get("id")),
                        "project_id": int(item.get("project")),
                        "entry_number": int(item.get("entry_number") or 0),
                        "worker_id": int(item.get("author")),
                        "entry_created_at": _to_utc(item.get("entry_created_at")),
                        "winner": bool(item.get("winner")),
                        "finalist": bool(item.get("finalist")),
                        "withdrawn": bool(item.get("withdrawn")),
                        "eliminated": bool(item.get("eliminated")),
                        "award_value": _float_or_zero(item.get("award_value")),
                        "tip_value": _float_or_zero(item.get("tip_value")),
                        "score": _entry_score(item),
                    }
                )
    entries = pd.DataFrame(rows, columns=ENTRY_COLUMNS)
    if entries.empty:
        return entries
    entries = entries.drop_duplicates("entry_id")
    return entries.sort_values(["entry_created_at", "entry_id"]).reset_index(drop=True)


def _load_quality(data_dir: Path) -> pd.DataFrame:
    quality = pd.read_csv(data_dir / "worker_quality.csv")
    quality = quality.rename(columns={"worker_id": "worker_id", "worker_quality": "worker_quality_raw"})
    quality["worker_id"] = quality["worker_id"].astype(int)
    quality["worker_quality_raw"] = pd.to_numeric(quality["worker_quality_raw"], errors="coerce")
    return quality


def _build_workers(entries: pd.DataFrame, raw_quality: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    workers = pd.DataFrame({"worker_id": sorted(entries["worker_id"].unique())})
    workers = workers.merge(raw_quality, on="worker_id", how="left")
    valid = workers["worker_quality_raw"].where(workers["worker_quality_raw"] >= 0)
    train_ids = train_worker_ids(events)
    train_valid = workers.loc[workers["worker_id"].isin(train_ids), "worker_quality_raw"]
    train_valid = train_valid[train_valid >= 0]
    fill_raw = float(train_valid.mean()) if not train_valid.empty else float(valid.mean())
    if math.isnan(fill_raw):
        fill_raw = 0.0
    workers["worker_quality_raw"] = valid.fillna(fill_raw)
    workers["worker_quality"] = workers["worker_quality_raw"] / 100.0
    workers["quality_filled_from_train_mean"] = (
        raw_quality.set_index("worker_id")
        .reindex(workers["worker_id"])["worker_quality_raw"]
        .lt(0)
        .fillna(True)
        .to_numpy()
    )
    return workers.sort_values("worker_id").reset_index(drop=True)


def _build_events(entries: pd.DataFrame) -> pd.DataFrame:
    events = entries.copy()
    events = events.sort_values(["entry_created_at", "entry_id"]).reset_index(drop=True)
    events.insert(0, "event_id", range(len(events)))
    return events


def build_processed_tables(
    data_dir: str | Path,
    output_dir: str | Path,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> ProcessedPaths:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_list = _read_project_list(data_dir / "project_list.csv")
    projects = _load_projects(data_dir, project_list)
    project_ids = set(projects["project_id"].astype(int))
    entries = _load_entries(data_dir, project_ids)
    events = temporal_split(_build_events(entries), ratios=split_ratios)
    workers = _build_workers(entries, _load_quality(data_dir), events)

    paths = ProcessedPaths(
        projects_path=output_dir / "projects.parquet",
        entries_path=output_dir / "entries.parquet",
        workers_path=output_dir / "workers.parquet",
        events_path=output_dir / "events.parquet",
        splits_path=output_dir / "splits.json",
    )
    projects.to_parquet(paths.projects_path, index=False)
    entries.to_parquet(paths.entries_path, index=False)
    workers.to_parquet(paths.workers_path, index=False)
    events.to_parquet(paths.events_path, index=False)
    write_splits(events, paths.splits_path)
    return paths
