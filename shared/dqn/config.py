from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DQNConfig:
    artifact_dir: Path | str = Path("artifacts/processed")
    output_dir: Path | str = Path("outputs")
    experiment_name: str = "dqn"
    reward_type: str = "combined"
    network_type: str = "dqn"
    replay_type: str = "uniform"
    double_dqn: bool = False
    candidate_k: int = 20
    alpha: float = 0.5
    train_steps: int = 50_000
    eval_max_steps: int | None = 5_000
    batch_size: int = 64
    min_replay_size: int = 1_000
    replay_capacity: int = 50_000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    priority_epsilon: float = 1e-6
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.reward_type not in {"worker", "requester", "combined"}:
            raise ValueError("reward_type must be worker, requester, or combined")
        if self.network_type not in {"dqn", "dueling"}:
            raise ValueError("network_type must be dqn or dueling")
        if self.replay_type not in {"uniform", "prioritized"}:
            raise ValueError("replay_type must be uniform or prioritized")
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.artifact_dir = Path(self.artifact_dir)
        self.output_dir = Path(self.output_dir)

    def epsilon_at(self, step: int) -> float:
        if step >= self.epsilon_decay_steps:
            return self.epsilon_end
        ratio = max(step, 0) / max(self.epsilon_decay_steps, 1)
        return self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)

    def beta_at(self, step: int) -> float:
        if step >= self.train_steps:
            return self.per_beta_end
        ratio = max(step, 0) / max(self.train_steps, 1)
        return self.per_beta_start + ratio * (self.per_beta_end - self.per_beta_start)
