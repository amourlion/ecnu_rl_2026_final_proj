# Exp3 Vanilla DQN - Worker Reward

This experiment uses the shared DQN training skeleton with the standard DQN
network and uniform replay. It optimizes the worker-side reward, so it is the
main experiment for answering how reinforcement learning can maximize
participant benefit.

Run on Apple Silicon with uv:

```bash
cd exps/dqn_worker_reward
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python run.py
```

This experiment uses `device="cpu"` by default on Apple Silicon. The workload is
a small MLP with many replay-buffer samples, and local benchmarking showed the
native M-series CPU is much faster than MPS for this project.

Configuration:

- `reward_type="worker"`
- `network_type="dqn"`
- `replay_type="uniform"`
- `double_dqn=False`
- `candidate_k=20`
- `train_steps=133_584`
- `eval_max_steps=None`
- `batch_size=64`
- `learning_rate=5e-4`
- `epsilon_decay_steps=80_000`
- `seed=42`
- `device="cpu"`

Outputs:

- `outputs/v*/metrics.csv`
- `outputs/v*/training_curve.csv`
- `outputs/v*/training_curve.png` if matplotlib is available
- `outputs/v*/result_summary.json`
