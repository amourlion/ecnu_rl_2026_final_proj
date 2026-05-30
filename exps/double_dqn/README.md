# Exp5 Double DQN

This experiment uses the shared DQN training skeleton with the standard DQN
network, uniform replay, and Double DQN target selection. It optimizes the
combined reward and is directly comparable with Exp6 Dueling DQN and Exp7
Prioritized Replay DQN.

Run on Apple Silicon with uv:

```bash
cd exps/double_dqn
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python run.py
```

This experiment uses `device="cpu"` by default on Apple Silicon. The workload is
a small MLP with many replay-buffer samples, and local benchmarking showed the
native M-series CPU is much faster than MPS for this project.

Configuration:

- `reward_type="combined"`
- `network_type="dqn"`
- `replay_type="uniform"`
- `double_dqn=True`
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
