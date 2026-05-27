from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shared.dqn.agent import DQNAgent
from shared.dqn.config import DQNConfig
from shared.envs import CrowdsourcingRecEnv
from shared.metrics import evaluate_agent


def train_dqn(config: DQNConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env = CrowdsourcingRecEnv.from_artifacts(
        config.artifact_dir,
        split="train",
        candidate_k=config.candidate_k,
        reward_type=config.reward_type,
        alpha=config.alpha,
    )
    agent = DQNAgent(config)
    state = train_env.reset()
    done = bool(state.get("done", False))
    curve_rows = []

    for step in range(config.train_steps):
        if done:
            state = train_env.reset()
            done = bool(state.get("done", False))
        candidates = train_env.get_candidates()
        epsilon = config.epsilon_at(step)
        action = agent.act(state, candidates, epsilon=epsilon)
        next_state, reward, done, info = train_env.step(action)
        next_candidates = (
            train_env.get_candidates()
            if not done and not next_state.get("done", False)
            else candidates.iloc[0:0].copy()
        )
        agent.push_transition(
            state,
            candidates,
            action,
            reward,
            next_state,
            next_candidates,
            done,
        )
        loss = agent.optimize(step)
        if step % config.target_update_interval == 0:
            agent.sync_target()
        curve_rows.append(
            {
                "step": step,
                "reward": float(reward),
                "loss": loss,
                "epsilon": float(epsilon),
                "hit": bool(info["hit"]),
            }
        )
        state = next_state

    metrics = _evaluate_splits(agent, config)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(output_dir / "training_curve.csv", index=False)
    _write_training_curve_png(curve, output_dir / "training_curve.png")
    summary = {
        "experiment_name": config.experiment_name,
        "reward_type": config.reward_type,
        "network_type": config.network_type,
        "replay_type": config.replay_type,
        "double_dqn": config.double_dqn,
        "candidate_k": config.candidate_k,
        "train_steps": config.train_steps,
        "device": str(agent.device),
        "metrics": metrics,
    }
    with (output_dir / "result_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def _evaluate_splits(agent: DQNAgent, config: DQNConfig) -> dict[str, float]:
    rows = {}
    for split in ("valid", "test"):
        env = CrowdsourcingRecEnv.from_artifacts(
            config.artifact_dir,
            split=split,
            candidate_k=config.candidate_k,
            reward_type=config.reward_type,
            alpha=config.alpha,
        )
        metrics = evaluate_agent(agent, env, max_steps=config.eval_max_steps)
        for key, value in metrics.items():
            rows[f"{split}_{key}"] = value
    return rows


def _write_training_curve_png(curve: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if curve.empty:
        return
    rolling_reward = curve["reward"].rolling(window=200, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve["step"], rolling_reward)
    ax.set_xlabel("step")
    ax.set_ylabel("rolling reward")
    ax.set_title("DQN training curve")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
