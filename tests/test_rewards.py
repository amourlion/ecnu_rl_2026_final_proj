from __future__ import annotations

from shared.envs.rewards import compute_rewards


def test_miss_reward_is_shaped_by_project_similarity() -> None:
    unrelated = compute_rewards(
        hit=False,
        score=0.0,
        winner=False,
        finalist=False,
        withdrawn=False,
        award_value=0.0,
        tip_value=0.0,
        worker_quality=0.8,
        same_category=False,
        same_sub_category=False,
        same_industry=False,
    )
    similar = compute_rewards(
        hit=False,
        score=0.0,
        winner=False,
        finalist=False,
        withdrawn=False,
        award_value=0.0,
        tip_value=0.0,
        worker_quality=0.8,
        same_category=True,
        same_sub_category=True,
        same_industry=True,
    )

    assert similar["worker_reward"] > unrelated["worker_reward"]
    assert similar["requester_reward"] > unrelated["requester_reward"]
    assert similar["combined_reward"] > unrelated["combined_reward"]
    assert similar["worker_reward"] < 0.0
    assert similar["requester_reward"] < 0.0
