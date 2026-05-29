import json
from pathlib import Path

import pandas as pd

from shared.data_utils.preprocess import build_processed_tables
from shared.data_utils.split import temporal_split
from shared.envs.recommendation_env import CrowdsourcingRecEnv
from shared.metrics.evaluator import evaluate_agent


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_raw_data(root: Path) -> None:
    (root / "project").mkdir(parents=True)
    (root / "entry").mkdir(parents=True)
    (root / "project_list.csv").write_text("1,2\n2,1\n", encoding="utf-8")
    (root / "worker_quality.csv").write_text(
        "worker_id,worker_quality\n10,80\n20,-1\n30,60\n",
        encoding="utf-8",
    )

    base_project = {
        "sub_category": 2,
        "category": 7,
        "industry": "software",
        "status": "completed",
        "average_score": 3.0,
        "total_awards": 500.0,
        "creative_count": 4,
        "client_feedback": 90.0,
        "entry_count": 2,
    }
    p1 = {
        **base_project,
        "id": 1,
        "start_date": "2020-01-01T00:00:00Z",
        "deadline": "2020-01-10T00:00:00Z",
    }
    p2 = {
        **base_project,
        "id": 2,
        "industry": None,
        "entry_count": 1,
        "start_date": "2020-01-01T00:00:00Z",
        "deadline": "2020-01-06T00:00:00Z",
    }
    write_json(root / "project" / "project_1.txt", p1)
    write_json(root / "project" / "project_2.txt", p2)

    write_json(
        root / "entry" / "entry_1_0.txt",
        {
            "limit": 24,
            "count": 2,
            "results": [
                {
                    "id": 101,
                    "project": 1,
                    "entry_number": 1,
                    "author": 10,
                    "entry_created_at": "2020-01-02T00:00:00+00:00",
                    "winner": False,
                    "finalist": False,
                    "withdrawn": False,
                    "eliminated": False,
                    "award_value": None,
                    "tip_value": 0.0,
                    "revisions": [{"score": 2}],
                },
                {
                    "id": 102,
                    "project": 1,
                    "entry_number": 2,
                    "author": 20,
                    "entry_created_at": "2020-01-03T00:00:00+00:00",
                    "winner": True,
                    "finalist": True,
                    "withdrawn": False,
                    "eliminated": False,
                    "award_value": 500.0,
                    "tip_value": 0.0,
                    "revisions": [{"score": 4}],
                },
            ],
        },
    )
    write_json(
        root / "entry" / "entry_2_0.txt",
        {
            "limit": 24,
            "count": 1,
            "results": [
                {
                    "id": 201,
                    "project": 2,
                    "entry_number": 1,
                    "author": 30,
                    "entry_created_at": "2020-01-05T00:00:00+00:00",
                    "winner": False,
                    "finalist": False,
                    "withdrawn": True,
                    "eliminated": False,
                    "award_value": None,
                    "tip_value": 0.0,
                    "revisions": [{"score": 1}],
                }
            ],
        },
    )


def test_build_processed_tables_writes_parquet_and_fills_quality_from_train_mean(tmp_path):
    raw_dir = tmp_path / "data"
    out_dir = tmp_path / "artifacts"
    make_raw_data(raw_dir)

    result = build_processed_tables(raw_dir, out_dir, split_ratios=(0.67, 0.0, 0.33))

    assert result.projects_path.exists()
    assert result.entries_path.exists()
    assert result.workers_path.exists()
    assert result.events_path.exists()
    assert result.splits_path.exists()

    projects = pd.read_parquet(result.projects_path)
    workers = pd.read_parquet(result.workers_path)
    events = pd.read_parquet(result.events_path)

    assert set(projects["project_id"]) == {1, 2}
    assert projects.loc[projects["project_id"] == 2, "industry"].item() == "unknown"
    assert list(events["event_id"]) == [0, 1, 2]
    assert workers.loc[workers["worker_id"] == 20, "worker_quality"].item() == 0.8


def test_temporal_split_preserves_order_and_split_labels():
    events = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "entry_created_at": pd.to_datetime(
                [
                    "2020-01-02T00:00:00Z",
                    "2020-01-01T00:00:00Z",
                    "2020-01-04T00:00:00Z",
                    "2020-01-03T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    split = temporal_split(events, ratios=(0.5, 0.25, 0.25))

    assert list(split.sort_values("entry_created_at")["split"]) == [
        "train",
        "train",
        "valid",
        "test",
    ]


def test_environment_returns_candidates_from_active_projects_and_steps(tmp_path):
    raw_dir = tmp_path / "data"
    out_dir = tmp_path / "artifacts"
    make_raw_data(raw_dir)
    build_processed_tables(raw_dir, out_dir, split_ratios=(1.0, 0.0, 0.0))

    env = CrowdsourcingRecEnv.from_artifacts(
        out_dir,
        split="train",
        candidate_k=2,
        reward_type="worker",
    )
    state = env.reset()
    candidates = env.get_candidates()

    assert state["worker_id"] == 10
    assert set(candidates["project_id"]).issubset({1, 2})
    assert candidates["is_active"].all()

    next_state, reward, done, info = env.step(0)

    assert isinstance(next_state, dict)
    assert isinstance(reward, float)
    assert done is False
    assert info["recommended_project_id"] in set(candidates["project_id"])
    assert "worker_reward" in info
    assert "requester_reward" in info


def test_environment_keeps_true_project_when_candidate_pool_is_full():
    projects = pd.DataFrame(
        {
            "project_id": [1, 2, 3],
            "category": [1, 1, 1],
            "sub_category": [1, 1, 1],
            "industry": [1, 1, 1],
            "start_date": pd.to_datetime(["2020-01-01"] * 3, utc=True),
            "deadline": pd.to_datetime(["2020-01-10"] * 3, utc=True),
            "total_awards": [300.0, 200.0, 100.0],
            "average_score": [5.0, 4.0, 3.0],
            "creative_count": [1, 1, 1],
            "entry_count": [1, 1, 1],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [0],
            "worker_id": [10],
            "project_id": [3],
            "entry_created_at": pd.to_datetime(["2020-01-02"], utc=True),
            "split": ["train"],
            "score": [3.0],
            "winner": [False],
            "finalist": [False],
            "withdrawn": [False],
            "award_value": [0.0],
            "tip_value": [0.0],
        }
    )
    workers = pd.DataFrame({"worker_id": [10], "worker_quality": [0.8]})
    env = CrowdsourcingRecEnv(
        projects=projects,
        entries=events,
        workers=workers,
        events=events,
        split="train",
        candidate_k=2,
    )

    candidates = env.get_candidates()

    assert len(candidates) == 2
    assert 3 in set(candidates["project_id"])


def test_environment_shapes_miss_reward_by_project_similarity():
    projects = pd.DataFrame(
        {
            "project_id": [1, 2, 3],
            "category": [1, 1, 3],
            "sub_category": [10, 10, 30],
            "industry": [100, 100, 300],
            "start_date": pd.to_datetime(["2020-01-01"] * 3, utc=True),
            "deadline": pd.to_datetime(["2020-01-10"] * 3, utc=True),
            "total_awards": [300.0, 200.0, 100.0],
            "average_score": [5.0, 4.0, 3.0],
            "creative_count": [1, 1, 1],
            "entry_count": [1, 1, 1],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [0],
            "worker_id": [10],
            "project_id": [1],
            "entry_created_at": pd.to_datetime(["2020-01-02"], utc=True),
            "split": ["train"],
            "score": [3.0],
            "winner": [False],
            "finalist": [False],
            "withdrawn": [False],
            "award_value": [0.0],
            "tip_value": [0.0],
        }
    )
    workers = pd.DataFrame({"worker_id": [10], "worker_quality": [0.8]})

    similar_env = CrowdsourcingRecEnv(
        projects=projects,
        entries=events,
        workers=workers,
        events=events,
        split="train",
        candidate_k=3,
        reward_type="combined",
    )
    unrelated_env = CrowdsourcingRecEnv(
        projects=projects,
        entries=events,
        workers=workers,
        events=events,
        split="train",
        candidate_k=3,
        reward_type="combined",
    )

    similar_candidates = similar_env.get_candidates()
    unrelated_candidates = unrelated_env.get_candidates()
    similar_action = int(similar_candidates.index[similar_candidates["project_id"] == 2][0])
    unrelated_action = int(unrelated_candidates.index[unrelated_candidates["project_id"] == 3][0])

    _, similar_reward, _, similar_info = similar_env.step(similar_action)
    _, unrelated_reward, _, unrelated_info = unrelated_env.step(unrelated_action)

    assert similar_info["hit"] is False
    assert unrelated_info["hit"] is False
    assert similar_reward > unrelated_reward


def test_combined_diversity_reward_penalizes_repeated_project_recommendations():
    projects = pd.DataFrame(
        {
            "project_id": [1, 2],
            "category": [1, 1],
            "sub_category": [10, 10],
            "industry": [100, 100],
            "start_date": pd.to_datetime(["2020-01-01"] * 2, utc=True),
            "deadline": pd.to_datetime(["2020-01-10"] * 2, utc=True),
            "total_awards": [300.0, 200.0],
            "average_score": [5.0, 4.0],
            "creative_count": [1, 1],
            "entry_count": [1, 1],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [0, 1],
            "worker_id": [10, 10],
            "project_id": [1, 1],
            "entry_created_at": pd.to_datetime(
                ["2020-01-02", "2020-01-03"],
                utc=True,
            ),
            "split": ["train", "train"],
            "score": [3.0, 3.0],
            "winner": [False, False],
            "finalist": [False, False],
            "withdrawn": [False, False],
            "award_value": [0.0, 0.0],
            "tip_value": [0.0, 0.0],
        }
    )
    workers = pd.DataFrame({"worker_id": [10], "worker_quality": [0.8]})
    env = CrowdsourcingRecEnv(
        projects=projects,
        entries=events,
        workers=workers,
        events=events,
        split="train",
        candidate_k=2,
        reward_type="combined_diversity",
        lambda_repeat=0.02,
    )

    first_action = int(env.get_candidates().index[env.get_candidates()["project_id"] == 1][0])
    _, first_reward, _, first_info = env.step(first_action)
    second_action = int(env.get_candidates().index[env.get_candidates()["project_id"] == 1][0])
    _, second_reward, _, second_info = env.step(second_action)

    assert first_info["repeat_penalty"] == 0.0
    assert second_info["repeat_penalty"] == 1.0
    assert first_reward == first_info["combined_reward"]
    assert second_reward == second_info["combined_reward"] - 0.02
    assert second_reward < second_info["combined_reward"]


def test_evaluate_agent_outputs_common_metrics(tmp_path):
    raw_dir = tmp_path / "data"
    out_dir = tmp_path / "artifacts"
    make_raw_data(raw_dir)
    build_processed_tables(raw_dir, out_dir, split_ratios=(1.0, 0.0, 0.0))
    env = CrowdsourcingRecEnv.from_artifacts(out_dir, split="train", candidate_k=2)

    metrics = evaluate_agent("heuristic", env)

    expected = {
        "avg_worker_reward",
        "avg_requester_reward",
        "avg_combined_reward",
        "hitrate_at_1",
        "avg_score",
        "winner_rate",
        "finalist_rate",
        "withdrawn_rate",
        "avg_worker_quality",
        "project_coverage",
        "category_diversity",
    }
    assert expected.issubset(metrics.keys())
