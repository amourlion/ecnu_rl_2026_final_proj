from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def test_dqn_config_builds_dueling_and_prioritized_variants() -> None:
    from shared.dqn import DQNConfig

    dueling = DQNConfig(
        experiment_name="dueling_dqn",
        reward_type="combined",
        network_type="dueling",
        replay_type="uniform",
    )
    prioritized = DQNConfig(
        experiment_name="prioritized_replay_dqn",
        reward_type="combined",
        network_type="dqn",
        replay_type="prioritized",
    )

    assert dueling.network_type == "dueling"
    assert dueling.double_dqn is False
    assert prioritized.replay_type == "prioritized"


def test_feature_encoder_returns_state_and_candidate_tensors() -> None:
    from shared.dqn import FeatureEncoder

    state = {
        "worker_id": 101,
        "current_time": pd.Timestamp("2021-01-02T03:00:00Z"),
    }
    candidates = pd.DataFrame(
        {
            "project_id": [1, 2],
            "category": [3, 4],
            "sub_category": [10, 20],
            "industry": [7, 8],
            "total_awards": [100.0, 500.0],
            "average_score": [3.5, 4.5],
            "creative_count": [2, 9],
            "entry_count": [10, 50],
            "category_match": [1.0, 0.0],
            "remaining_hours": [12.0, 48.0],
            "competition": [10.0, 50.0],
            "heuristic_score": [2.5, 1.5],
        }
    )

    encoded = FeatureEncoder().encode(state, candidates)

    assert encoded.state.shape == (1, encoded.state_dim)
    assert encoded.candidates.shape == (2, encoded.candidate_dim)
    assert torch.isfinite(encoded.state).all()
    assert torch.isfinite(encoded.candidates).all()


def test_dueling_network_outputs_one_q_value_per_candidate() -> None:
    from shared.dqn import DuelingDQNNet

    model = DuelingDQNNet(state_dim=4, candidate_dim=6, hidden_dim=16)
    state = torch.zeros((1, 4), dtype=torch.float32)
    candidates = torch.zeros((5, 6), dtype=torch.float32)

    q_values = model(state, candidates)

    assert q_values.shape == (5,)


def test_prioritized_replay_samples_and_updates_priorities() -> None:
    from shared.dqn import PrioritizedReplayBuffer, Transition

    buffer = PrioritizedReplayBuffer(capacity=8, alpha=0.6)
    for i in range(4):
        buffer.push(
            Transition(
                state=torch.zeros((1, 2)),
                candidates=torch.zeros((3, 2)),
                action=0,
                reward=float(i),
                next_state=torch.zeros((1, 2)),
                next_candidates=torch.zeros((3, 2)),
                done=False,
            )
        )

    batch = buffer.sample(batch_size=3, beta=0.4)
    buffer.update_priorities(batch.indices, torch.tensor([0.5, 1.0, 2.0]))

    assert len(batch.transitions) == 3
    assert batch.weights.shape == (3,)
    assert all(priority > 0 for priority in buffer.priorities[: len(buffer)])


def test_train_dqn_smoke_writes_metrics(tmp_path: Path) -> None:
    from shared.dqn import DQNConfig, train_dqn

    config = DQNConfig(
        artifact_dir=Path("artifacts/processed"),
        output_dir=tmp_path,
        experiment_name="smoke_dqn",
        reward_type="combined",
        train_steps=4,
        eval_max_steps=3,
        batch_size=2,
        min_replay_size=1,
        candidate_k=5,
        hidden_dim=16,
        target_update_interval=2,
        seed=7,
    )

    summary = train_dqn(config)

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "result_summary.json").exists()
    assert (tmp_path / "training_curve.csv").exists()
    assert summary["experiment_name"] == "smoke_dqn"
