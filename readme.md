# ECNU RL 2026 Final Project

本项目用于完成强化学习课程期末作业：将强化学习应用到众包系统的任务推荐中。当 worker 进入平台时，系统只推荐一个 project；我们需要设计推荐策略，分别最大化参与者利益和请求者利益，并用 DQN 系列模型完成实验验证。

项目会采用多人协作方式推进。为了减少环境冲突和实验互相影响，每个实验都放在 `exps/[expname]` 下，并维护自己独立的 Python 虚拟环境、依赖文件、训练脚本、结果输出和实验说明。

## 项目目的

核心问题：

- 如何将强化学习应用到众包任务推荐中，并最大化参与者利益？
- 如何将强化学习应用到众包任务推荐中，并最大化请求者利益？

主要工作：

- 读取并清洗 crowdsourcing 数据。
- 构造 worker、project、时间、历史交互等特征。
- 将任务推荐建模为 MDP，定义 state、action、reward、transition。
- 按时间顺序模拟 worker 到达和任务推荐。
- 实现 baseline、Q-learning、SARSA、DQN 系列、Actor-Critic/A2C 等实验。
- 对比不同方法在参与者收益、请求者收益、推荐命中率、覆盖率、多样性等指标上的表现。
- 输出实验报告和汇报材料。

## 目录结构

当前和计划中的目录结构如下：

```text
.
├── agents.md
├── readme.md
├── docs/
│   └── task.md
├── data/
│   ├── sample_read_data.py
│   ├── project_list.csv
│   ├── worker_quality.csv
│   ├── project/
│   └── entry/
├── exps/
│   ├── random_baseline/
│   │   ├── .venv/
│   │   ├── requirements.txt
│   │   ├── run.py
│   │   ├── README.md
│   │   └── outputs/
│   ├── dqn_worker_reward/
│   │   ├── .venv/
│   │   ├── requirements.txt
│   │   ├── run.py
│   │   ├── README.md
│   │   └── outputs/
│   └── ...
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.md
└── shared/
    ├── data_utils/
    ├── envs/
    ├── metrics/
    └── plotting/
```

说明：

- `docs/task.md`：作业原始要求。
- `agents.md`：项目执行计划、实验矩阵、建模思路和验证清单。
- `data/`：原始数据，不建议直接修改。
- `exps/`：每个实验的独立工作区，实验之间依赖隔离。
- `shared/`：可复用公共代码，例如数据读取、环境模拟、指标计算、绘图工具。
- `reports/`：最终报告、图表、实验结果汇总。

## 实验目录规范

每个实验必须放在：

```text
exps/[expname]/
```

推荐命名：

- `random_baseline`
- `heuristic_baseline`
- `q_learning`
- `sarsa`
- `dqn_worker_reward`
- `dqn_requester_reward`
- `double_dqn`
- `dueling_dqn`
- `prioritized_replay_dqn`
- `a2c`

每个实验目录建议包含：

```text
exps/[expname]/
├── .venv/
├── requirements.txt
├── README.md
├── run.py
├── config.yaml
├── src/
├── outputs/
│   ├── metrics.csv
│   ├── training_curve.png
│   └── result_summary.json
└── notes.md
```

要求：

- `.venv/` 只在本地使用，不提交到 git。
- `requirements.txt` 记录该实验的完整依赖。
- `README.md` 说明该实验的目的、奖励函数、运行命令和输出文件。
- `outputs/` 保存结果，较小的指标表和图可以提交；大模型权重和大缓存文件不要提交。
- 不同实验之间不要互相修改代码，公共逻辑放到 `shared/`。

## 虚拟环境约定

每个实验都使用独立虚拟环境：

```bash
cd exps/[expname]
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

如果实验需要 PyTorch、TensorFlow、stable-baselines3 或其他较重依赖，只安装在该实验自己的 `.venv` 中，避免影响其他成员。

## 推荐实验矩阵

计划完成 8 组正式实验和 2 个 baseline：

| 编号 | 方法 | 奖励目标 | 说明 |
| --- | --- | --- | --- |
| Baseline A | Random | 无学习 | 随机推荐 |
| Baseline B | Heuristic Ranking | 启发式综合分 | 按奖金、剩余时间、热度、平均分排序 |
| Exp 1 | Tabular Q-learning | 综合奖励 | 传统 off-policy RL |
| Exp 2 | SARSA | 综合奖励 | 传统 on-policy RL |
| Exp 3 | Vanilla DQN | 参与者奖励 | 最大化参与者利益 |
| Exp 4 | Vanilla DQN | 请求者奖励 | 最大化请求者利益 |
| Exp 5 | Double DQN | 综合奖励 | 降低 Q 值过估计 |
| Exp 6 | Dueling DQN | 综合奖励 | 拆分 V(s) 和 A(s,a) |
| Exp 7 | Prioritized Replay DQN | 综合奖励 | 优先学习高 TD error 样本 |
| Exp 8 | Actor-Critic / A2C | 综合奖励 | 策略梯度扩展 |

如果时间不足，优先完成：

1. Random baseline
2. Heuristic baseline
3. Vanilla DQN with worker reward
4. Vanilla DQN with requester reward
5. Double DQN
6. Dueling DQN

## 输出与报告要求

每个实验至少输出：

- `metrics.csv`：统一指标表。
- `training_curve.png`：训练奖励曲线。
- `result_summary.json`：关键参数和最终测试结果。
- `README.md` 或 `notes.md`：实验说明和结论。

最终报告需要回答：

- 参与者利益如何定义，哪个方法表现最好。
- 请求者利益如何定义，哪个方法表现最好。
- DQN 系列相比传统 RL 和启发式方法是否有提升。
- 不同 reward 设计对推荐结果有什么影响。
- 动态时间模拟是否比静态排序更合理。
