from __future__ import annotations

import math


def _award_scale(value: float) -> float:
    return math.log1p(max(value, 0.0)) / math.log1p(5000.0)


def compute_rewards(
    hit: bool,
    score: float,
    winner: bool,
    finalist: bool,
    withdrawn: bool,
    award_value: float,
    tip_value: float,
    worker_quality: float,
    alpha: float = 0.5,
) -> dict[str, float]:
    if not hit:
        worker_reward = -0.1
        requester_reward = -0.05
    else:
        score_norm = max(score, 0.0) / 5.0
        money_norm = _award_scale(award_value + tip_value)
        worker_reward = 1.0 + score_norm + money_norm
        worker_reward += 1.0 if winner else 0.0
        worker_reward += 0.5 if finalist else 0.0
        worker_reward -= 0.5 if withdrawn else 0.0

        requester_reward = max(worker_quality, 0.0)
        requester_reward += 0.3 if not withdrawn else -0.5
        requester_reward += score_norm
        requester_reward += 1.0 if winner else 0.0
        requester_reward += 0.5 if finalist else 0.0

    combined = alpha * worker_reward + (1.0 - alpha) * requester_reward
    return {
        "worker_reward": float(worker_reward),
        "requester_reward": float(requester_reward),
        "combined_reward": float(combined),
    }

