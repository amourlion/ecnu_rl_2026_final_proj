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
    diversity_ablation = DQNConfig(
        experiment_name="combined_diversity_ablation",
        reward_type="combined_diversity",
        network_type="dqn",
        replay_type="uniform",
    )

    assert dueling.network_type == "dueling"
    assert dueling.double_dqn is False
    assert prioritized.replay_type == "prioritized"
    assert diversity_ablation.reward_type == "combined_diversity"


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


def test_feature_encoder_preserves_legacy_non_numeric_and_missing_column_behavior() -> None:
    from shared.dqn import FeatureEncoder

    encoder = FeatureEncoder()
    candidates = pd.DataFrame(
        {
            "project_id": [1],
            "category": [3],
            "sub_category": [10],
            "industry": ["entertainment-and-sports"],
        }
    )

    encoded = encoder._encode_candidates_batch(candidates).cpu()

    assert encoded.shape == (1, encoder.candidate_dim)
    assert encoded[0, 3].item() == 0.0
    assert encoded[0, 7].item() == 0.0
    assert torch.isfinite(encoded).all()


def test_dueling_network_outputs_one_q_value_per_candidate() -> None:
    from shared.dqn import DuelingDQNNet

    model = DuelingDQNNet(state_dim=4, candidate_dim=6, hidden_dim=16)
    state = torch.zeros((1, 4), dtype=torch.float32)
    candidates = torch.zeros((5, 6), dtype=torch.float32)

    q_values = model(state, candidates)

    assert q_values.shape == (5,)


def test_double_dqn_uses_one_batched_online_next_forward(monkeypatch) -> None:
    from shared.dqn import DQNAgent, DQNConfig, Transition

    config = DQNConfig(
        double_dqn=True,
        batch_size=2,
        min_replay_size=1,
        hidden_dim=8,
        device="cpu",
    )
    agent = DQNAgent(config)
    state = torch.zeros((1, agent.encoder.state_dim), dtype=torch.float32)
    candidates = torch.zeros((2, agent.encoder.candidate_dim), dtype=torch.float32)
    next_candidates = torch.zeros((3, agent.encoder.candidate_dim), dtype=torch.float32)
    for reward in (1.0, 2.0):
        agent.replay.push(
            Transition(
                state=state,
                candidates=candidates,
                action=0,
                reward=reward,
                next_state=state,
                next_candidates=next_candidates,
                done=False,
            )
        )

    original_forward_batch = agent.policy_net.forward_batch
    calls: list[int] = []

    def wrapped_forward_batch(states, candidates_list):
        calls.append(len(states))
        return original_forward_batch(states, candidates_list)

    monkeypatch.setattr(agent.policy_net, "forward_batch", wrapped_forward_batch)

    loss = agent.optimize(step=0)

    assert loss is not None
    assert calls == [2, 2]


def test_auto_device_prefers_mps_when_cuda_is_unavailable(monkeypatch) -> None:
    from shared.dqn import DQNAgent, DQNConfig

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    agent = DQNAgent.__new__(DQNAgent)

    assert agent._select_device("auto").type == "mps"


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
    output_dir = Path(summary["output_dir"])

    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "result_summary.json").exists()
    assert (output_dir / "training_curve.csv").exists()
    assert summary["experiment_name"] == "smoke_dqn"
