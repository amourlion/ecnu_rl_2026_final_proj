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

Outputs:

- `outputs/metrics.csv`
- `outputs/training_curve.csv`
- `outputs/training_curve.png` if matplotlib is available
- `outputs/result_summary.json`
