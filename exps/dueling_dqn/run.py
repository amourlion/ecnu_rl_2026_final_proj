from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dqn import DQNConfig, train_dqn


def main() -> None:
    exp_dir = Path(__file__).resolve().parent
    config = DQNConfig(
        artifact_dir=ROOT / "artifacts/processed",
        output_dir=exp_dir / "outputs",
        experiment_name=exp_dir.name,
        reward_type="combined",
        network_type="dueling",
        replay_type="uniform",
        double_dqn=False,
        candidate_k=20,
        train_steps=50_000,
        eval_max_steps=5_000,
        device="auto",
    )
    train_dqn(config)


if __name__ == "__main__":
    main()
