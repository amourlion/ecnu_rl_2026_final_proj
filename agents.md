# 强化学习众包任务推荐作业执行计划

## 1. 作业目标

本项目目标是把强化学习应用到众包任务推荐中：当一个 worker 进入平台时，系统只推荐一个 project，并通过推荐策略分别优化参与者和请求者的利益。

作业要求主线是 **DQN 系列模型**，因此实验设计以 DQN、Double DQN、Dueling DQN、Prioritized Replay DQN 为核心；同时加入 Q-learning、SARSA、Actor-Critic/A2C 作为对照或扩展，体现足够的实验工作量。

最终交付物：

- 一套可复现实验流程。
- 一组 5-10 个核心实验，推荐采用 8 组正式实验 + 2 个 baseline。
- 实验结果表格、训练曲线、指标对比图。
- 一份实验报告，分别回答“最大化参与者利益”和“最大化请求者利益”两个问题。

## 2. 数据与预处理任务

需要读取并整理以下数据：

- `data/project_list.csv`：project_id 与 project 总 entry 数。
- `data/worker_quality.csv`：worker 的质量分数。
- `data/project/`：project 详细信息，包括 category、sub_category、industry、start_date、deadline、entry_count、average_score、total_awards、creative_count、client_feedback 等。
- `data/entry/`：worker 对 project 的提交记录，包括 author、entry_created_at、winner、finalist、withdrawn、award_value、tip_value、revision score 等。

预处理步骤：

1. 读取全部 project 信息，过滤缺失严重或时间字段异常的数据。
2. 读取全部 entry 信息，按 `entry_created_at` 排序。
3. 合并 worker_quality，缺失质量分可用均值或 0 填充，并在报告中说明。
4. 生成 worker 历史画像：参与过的 category、平均得分、中奖率、入围率、平均奖金、平均质量。
5. 生成 project 动态画像：当前 entry 数、剩余时间、历史热度、平均得分、奖金、是否接近 deadline。
6. 按时间划分训练集、验证集、测试集，建议比例为 70% / 15% / 15%，避免未来信息泄露。

## 3. 强化学习环境设计

### 3.1 状态 State

每个状态表示“当前 worker 到达时，worker 与候选 project 池的上下文”。

建议包含以下特征：

- Worker 特征：worker_quality、历史参与次数、历史平均 score、历史 winner/finalist 比例、常参与 category/sub_category。
- Project 特征：category、sub_category、industry、total_awards、average_score、creative_count、当前 entry 数、历史热度。
- 时间特征：当前时间、project 已开放时长、距离 deadline 的剩余时间、是否在早期/中期/末期。
- 匹配特征：worker 历史偏好与 project category/sub_category/industry 的匹配程度。
- 候选池特征：当前可推荐 project 数量、候选 project 的平均奖金、平均剩余时间、平均竞争强度。

### 3.2 动作 Action

动作是在当前 worker 到达时，从候选 project 池中选择一个 project 推荐。

候选 project 需要满足：

- `start_date <= 当前时间 <= deadline`
- project 当前仍可参与
- project 在训练或测试时间窗口内可见

为控制动作空间，建议每次只保留 Top-K 候选 project，例如 K=20 或 K=50。候选池可按奖金、剩余时间、热度、category 匹配度进行初筛。

### 3.3 奖励 Reward

需要设计三类奖励，分别用于不同实验。

参与者利益奖励：

- 推荐后 worker 实际参与该 project：正奖励。
- entry revision score 更高：正奖励。
- finalist 或 winner：额外正奖励。
- award_value / tip_value 更高：额外正奖励。
- 推荐但无实际参与：负奖励或 0 奖励。

请求者利益奖励：

- 推荐带来高 quality worker：正奖励。
- worker 提交 entry 且未 withdrawn：正奖励。
- entry score 高、finalist、winner：更高正奖励。
- project 获得更多有效 entry：正奖励。
- 低质量或 withdrawn entry：低奖励或负奖励。

综合目标奖励：

```text
reward = alpha * worker_reward + (1 - alpha) * requester_reward
```

建议先使用 `alpha = 0.5`，再在附加实验或消融分析中尝试 `alpha = 0.3 / 0.7`。

### 3.4 状态转移 Next State

按时间顺序模拟 worker 到达：

1. 当前 worker 到达，生成候选 project 池。
2. agent 推荐一个 project。
3. 根据历史数据判断该 worker 是否实际参与该 project，并计算奖励。
4. 更新 worker/project 历史统计。
5. 进入下一个 worker 到达事件。

## 4. 实验矩阵

正式报告主表建议展示 8 组正式实验；baseline 单独放在对照区，总实验量控制在 10 组以内。

| 编号 | 方法 | 奖励目标 | 作用 |
| --- | --- | --- | --- |
| Baseline A | Random | 无学习 | 随机推荐，作为最低基准 |
| Baseline B | Heuristic Ranking | 启发式综合分 | 按奖金、剩余时间、历史热度、平均分排序 |
| Exp 1 | Tabular Q-learning | 综合奖励 | 传统 off-policy RL 对照，状态离散化 |
| Exp 2 | SARSA | 综合奖励 | 传统 on-policy RL 对照 |
| Exp 3 | Vanilla DQN | 参与者奖励 | 回答最大化参与者利益 |
| Exp 4 | Vanilla DQN | 请求者奖励 | 回答最大化请求者利益 |
| Exp 5 | Double DQN | 综合奖励 | 减少 Q 值过估计 |
| Exp 6 | Dueling DQN | 综合奖励 | 拆分 state value 与 action advantage |
| Exp 7 | Prioritized Replay DQN | 综合奖励 | 强化稀有高价值样本学习 |
| Exp 8 | Actor-Critic / A2C | 综合奖励 | 策略梯度扩展方法 |

优先级安排：

1. 必做：Baseline A、Baseline B、Vanilla DQN worker reward、Vanilla DQN requester reward。
2. 推荐做：Q-learning、Double DQN、Dueling DQN。
3. 加分做：SARSA、Prioritized Replay DQN、Actor-Critic/A2C。
4. 如果时间不足，至少保留 5 组：Baseline、Q-learning、DQN、Double DQN、Dueling DQN。

## 5. 模型实现要点

### 5.1 Q-learning / SARSA

- 将连续特征离散化，例如剩余时间分桶、奖金分桶、worker_quality 分桶。
- Q 表键可以设计为 `(worker_type, project_type, time_bucket, competition_bucket)`。
- Q-learning 使用 max next Q 更新。
- SARSA 使用实际下一动作的 Q 值更新。

### 5.2 DQN 系列

统一使用相同输入特征、候选池构造和评价指标，保证公平对比。

基础配置建议：

- 网络：MLP，2-3 层隐藏层。
- 输入：state-action pair 特征或 state + candidate project features。
- 输出：每个候选 project 的 Q 值。
- 训练：experience replay、target network、epsilon-greedy。
- 损失：MSE 或 Huber loss。

DQN 变体：

- Vanilla DQN：基础深度 Q 网络。
- Double DQN：online network 选动作，target network 估值。
- Dueling DQN：输出 `V(s)` 和 `A(s,a)`，再组合为 Q 值。
- Prioritized Replay DQN：按 TD error 优先采样 replay buffer。

### 5.3 Actor-Critic / A2C

- Actor 输出候选 project 的选择概率。
- Critic 估计当前 state value。
- 使用综合奖励训练。
- 作为扩展方法即可，不需要成为报告主结论。

## 6. 评价指标

所有方法都输出同一套指标。

参与者侧指标：

- 平均 worker reward。
- 推荐后命中真实参与 project 的 HitRate@1。
- 推荐 project 的平均 award_value / tip_value。
- 推荐 entry 的平均 score。
- winner/finalist 命中率。

请求者侧指标：

- 平均 requester reward。
- 推荐 worker 的平均 worker_quality。
- 有效 entry 数提升。
- withdrawn 比例降低情况。
- 高 score entry 比例。

推荐系统指标：

- NDCG@K 或 HitRate@K，若实现 Top-K 评估。
- project 覆盖率，避免只推荐少数热门项目。
- category / industry 多样性。
- 平均奖励训练曲线和验证集曲线。

## 7. 报告结构

实验报告建议按以下结构写：

1. 问题背景：众包任务推荐为什么适合强化学习。
2. MDP 建模：state、action、reward、transition 的定义。
3. 数据处理：数据字段、清洗、时间划分、特征工程。
4. 方法介绍：baseline、Q-learning、SARSA、DQN 系列、A2C。
5. 实验设置：训练参数、候选池大小、奖励权重、评价指标。
6. 实验结果：主结果表、训练曲线、参与者/请求者目标分别分析。
7. 消融与讨论：奖励函数差异、DQN 变体差异、动态推荐效果。
8. 结论：哪个方法最适合参与者目标，哪个方法最适合请求者目标，综合目标如何权衡。

## 8. 验证清单

实现完成后需要检查：

- project、entry、worker_quality 能正确读取和合并。
- 时间排序正确，没有使用未来信息。
- 当前候选 project 都满足 start/deadline 约束。
- 每次推荐只输出一个 project。
- 推荐 project 一定来自当前候选池。
- reward 数值稳定，没有大量 NaN 或极端异常值。
- 所有实验输出统一格式的 metrics 表。
- 所有训练曲线和结果图都能直接放进报告。

## 9. 默认假设

- `agents.md` 负责记录作业执行计划和实验矩阵，不直接放完整代码实现。
- 实验数量采用 8 组正式实验 + 2 个 baseline。
- DQN 系列是主线，Q-learning、SARSA、Actor-Critic/A2C 是对照或扩展。
- 若时间紧，优先保证 baseline、DQN 参与者目标、DQN 请求者目标、Double DQN、Dueling DQN 的完整结果。
