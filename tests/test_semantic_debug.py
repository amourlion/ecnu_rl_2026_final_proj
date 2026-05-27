from tests.test_shared_interfaces import make_raw_data

from shared.data_utils.preprocess import build_processed_tables
from shared.semantic_debug import (
    debug_environment_interface,
    debug_metrics_interface,
    debug_processed_artifacts,
)


def test_semantic_debug_validates_processed_env_and_metrics(tmp_path):
    raw_dir = tmp_path / "data"
    out_dir = tmp_path / "artifacts"
    make_raw_data(raw_dir)
    build_processed_tables(raw_dir, out_dir, split_ratios=(1.0, 0.0, 0.0))

    processed_report = debug_processed_artifacts(out_dir)
    env_report = debug_environment_interface(out_dir, split="train", candidate_k=2, steps=2)
    metrics_report = debug_metrics_interface(out_dir, split="train", candidate_k=2)

    assert processed_report["ok"] is True
    assert env_report["ok"] is True
    assert metrics_report["ok"] is True
    assert processed_report["projects"] == 2
    assert env_report["checked_steps"] == 2
    assert "hitrate_at_1" in metrics_report["metrics"]
