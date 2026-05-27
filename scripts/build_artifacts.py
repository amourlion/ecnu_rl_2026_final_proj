from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.data_utils.preprocess import build_processed_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shared parquet artifacts.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts/processed")
    args = parser.parse_args()
    paths = build_processed_tables(args.data_dir, args.output_dir)
    print(f"projects={paths.projects_path}")
    print(f"entries={paths.entries_path}")
    print(f"workers={paths.workers_path}")
    print(f"events={paths.events_path}")
    print(f"splits={paths.splits_path}")


if __name__ == "__main__":
    main()
