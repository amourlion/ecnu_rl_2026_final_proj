from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.semantic_debug import (
    debug_environment_interface,
    debug_metrics_interface,
    debug_processed_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic checks on shared interfaces.")
    parser.add_argument("--artifact-dir", default="artifacts/processed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    report = {
        "processed": debug_processed_artifacts(args.artifact_dir),
        "environment": debug_environment_interface(
            args.artifact_dir,
            split=args.split,
            candidate_k=args.candidate_k,
            steps=args.steps,
        ),
        "metrics": debug_metrics_interface(
            args.artifact_dir,
            split=args.split,
            candidate_k=args.candidate_k,
            steps=args.steps,
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
