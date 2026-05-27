# DQN Experiment Template

This template is for Exp3-Exp7. Copy this directory to a concrete experiment
directory and change only the config values in `run.py`.

Example:

```bash
cp -r exps/dqn_template exps/dueling_dqn
cd exps/dueling_dqn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
python run.py
```

Use a CUDA wheel compatible with the local NVIDIA driver. On Windows + WSL,
first confirm that `nvidia-smi` and `/dev/dxg` work inside WSL.

Variant settings:

| Experiment | `reward_type` | `network_type` | `replay_type` | `double_dqn` |
| --- | --- | --- | --- | --- |
| Exp3 DQN worker | `worker` | `dqn` | `uniform` | `False` |
| Exp4 DQN requester | `requester` | `dqn` | `uniform` | `False` |
| Exp5 Double DQN | `combined` | `dqn` | `uniform` | `True` |
| Exp6 Dueling DQN | `combined` | `dueling` | `uniform` | `False` |
| Exp7 PER DQN | `combined` | `dqn` | `prioritized` | `False` |

