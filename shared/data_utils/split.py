from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


SPLIT_NAMES = ("train", "valid", "test")


def temporal_split(
    events: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> pd.DataFrame:
    """Assign train/valid/test labels by chronological event order."""
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, valid, and test shares")
    if not 0.999 <= sum(ratios) <= 1.001:
        raise ValueError("split ratios must sum to 1.0")
    if "entry_created_at" not in events.columns:
        raise ValueError("events must contain entry_created_at")

    ordered = events.copy()
    ordered["entry_created_at"] = pd.to_datetime(ordered["entry_created_at"], utc=True)
    ordered = ordered.sort_values(["entry_created_at", "event_id"]).reset_index(drop=True)

    n_events = len(ordered)
    train_end = int(n_events * ratios[0])
    valid_end = train_end + int(n_events * ratios[1])
    if ratios[0] > 0 and train_end == 0 and n_events:
        train_end = 1
    if ratios[1] > 0 and valid_end == train_end and n_events - train_end > 1:
        valid_end += 1

    labels = []
    for pos in range(n_events):
        if pos < train_end:
            labels.append("train")
        elif pos < valid_end:
            labels.append("valid")
        else:
            labels.append("test")
    ordered["split"] = labels
    return ordered


def write_splits(events: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        split: events.loc[events["split"] == split, "event_id"].astype(int).tolist()
        for split in SPLIT_NAMES
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_ids(path: Path, split: str) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(int(event_id) for event_id in payload.get(split, []))


def train_worker_ids(events: pd.DataFrame) -> set[int]:
    if "split" not in events.columns:
        return set()
    return set(events.loc[events["split"] == "train", "worker_id"].astype(int))

