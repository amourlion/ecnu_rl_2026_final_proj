# Exp6 Dueling DQN

This experiment uses the shared DQN training skeleton with a dueling Q network.
It optimizes the combined reward and keeps the same artifacts, environment,
candidate pool, and evaluation metrics as the other DQN experiments.

Run:

```bash
cd exps/dueling_dqn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
python run.py
```

Configuration:

- `reward_type="combined"`
- `network_type="dueling"`
- `replay_type="uniform"`
- `double_dqn=False`
- `candidate_k=20`
- `train_steps=133_584`
- `eval_max_steps=None`
- `batch_size=64`
- `learning_rate=5e-4`
- `epsilon_decay_steps=80_000`
- `seed=42`

Formal setting:

- `train_steps=133_584` matches one full pass over the train split event timeline.
- `eval_max_steps=None` evaluates the full validation and test splits.
- `learning_rate=5e-4` is used for more stable long-horizon DQN training.
- `epsilon_decay_steps=80_000` keeps exploration active for about 60% of training.
- `seed=42` keeps the experiment reproducible and comparable with other DQN runs.

Outputs:

- `outputs/metrics.csv`
- `outputs/training_curve.csv`
- `outputs/training_curve.png` if matplotlib is available
- `outputs/result_summary.json`
