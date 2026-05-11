# AER (Adaptive Exploration Reward) 算法详解

## 概述

AER是一种用于数学推理强化学习的自适应探索奖励机制。它通过**动态权重**在**准确性奖励**和**探索奖励**之间进行平衡，鼓励模型在保持答案正确的同时，产生多样化的推理路径。

## 核心思想

```
最终奖励 = 准确性奖励 + 动态权重 × 探索奖励
```

- **准确性奖励**：鼓励模型给出正确答案
- **探索奖励**：鼓励模型产生多样化的推理路径
- **动态权重**：根据当前探索水平自动调整

---

## 算法流程

### 1. 数据组织

对于每个prompt，模型生成 `n` 个响应（由配置中的 `rollout_n` 控制）：

```python
# main_ppo.py 中设置
rollout.n = 4  # 每个prompt生成4个响应
```

同一prompt的所有响应共享同一个 `uid`，形成**一个组(group)**。

### 2. 准确性奖励计算

**代码位置**: `reward_manager.py:24-54`

```python
def math_accuracy_reward(solution: str, golden_answer: str) -> Dict[str, float | str]:
    extracted_golden_answer = parse("\\boxed{" + golden_answer + "}")
    extracted_answer = parse(solution[-512:])
    reward = float(verify(extracted_golden_answer[0], extracted_answer[0]))
    return {"score": reward, "acc": reward, "pred": str(extracted_answer[-1])}
```

**计算公式**:

$$r_{acc}(i) = \begin{cases} 1.0 & \text{if answer}_i = \text{ground\_truth} \\ 0.0 & \text{otherwise} \end{cases}$$

### 3. 探索奖励计算

#### 3.1 Token相似度计算

**代码位置**: `reward_manager.py:128-165`

首先计算同一组内响应之间的token相似度：

```python
def compute_token_similarity(data: DataProto) -> torch.Tensor:
    token_matrix = data.batch["responses"]  # shape: (n, seq_len)
    response_mask = data.batch["attention_mask"][:, -response_length:]

    # 计算每对响应的token匹配
    a = token_matrix.unsqueeze(1)  # (n, 1, l)
    b = token_matrix.unsqueeze(0)  # (1, n, l)
    pos_match = (a == b).float()  # (n, n, l)

    # 计算归一化因子
    valid_lengths = response_mask.sum(1).float()
    lengths_a = valid_lengths.unsqueeze(1)  # (n, 1)
    lengths_b = valid_lengths.unsqueeze(0)  # (1, n)
    norm_factor = torch.sqrt(lengths_a * lengths_b) + 1e-8

    # 计算相似度
    overlap_sim = (pos_match * valid_pos_mask.float()).sum(2) / norm_factor

    # 只保留同一组内的相似度
    ids = data.non_tensor_batch["uid"]
    _, inverse_indices = np.unique(ids, return_inverse=True)
    index_tensor = torch.tensor(inverse_indices, device=token_matrix.device, dtype=torch.long)
    group_mask = (index_tensor.unsqueeze(0) == index_tensor.unsqueeze(1))
    overlap_sim = overlap_sim * group_mask

    return overlap_sim  # shape: (n, n)
```

**计算公式**:

$$\text{sim}(i, j) = \begin{cases} \frac{\sum_{k} [token_i^k = token_j^k] \cdot mask_i^k \cdot mask_j^k}{\sqrt{L_i \cdot L_j}} & \text{if } uid_i = uid_j \\ 0 & \text{otherwise} \end{cases}$$

其中:
- $token_i^k$ 是响应 $i$ 在位置 $k$ 的token
- $mask_i^k$ 是响应 $i$ 在位置 $k$ 的注意力掩码
- $L_i = \sum_k mask_i^k$ 是响应 $i$ 的有效长度

#### 3.2 探索奖励

**代码位置**: `reward_manager.py:174-239`

```python
class AERRewardManager:
    def __call__(self, data: DataProto, ...):
        similarity = compute_token_similarity(data)
        similarity_sum = similarity.sum(-1)  # 对每个响应，计算与同组其他响应的总相似度

        for i in range(len(data)):
            # 探索奖励 = 1 / 总相似度
            exploration_reward = 1.0 / similarity_sum[i].item() if similarity_sum[i].item() > 0 else 0.0
            reward_tensor_exploration[i, valid_response_length - 1] = exploration_reward
```

**计算公式**:

$$\text{sim\_sum}(i) = \sum_{j} \text{sim}(i, j)$$

$$r_{exp}(i) = \begin{cases} \frac{1}{\text{sim\_sum}(i)} & \text{if } \text{sim\_sum}(i) > 0 \\ 0 & \text{otherwise} \end{cases}$$

**直观理解**:
- 同一组的响应越相似 → sim_sum越大 → 探索奖励越小（惩罚重复）
- 同一组的响应越不同 → sim_sum越小 → 探索奖励越大（鼓励多样性）

### 4. 动态权重更新

**代码位置**: `aer_ray_trainer.py:124-133`

```python
def _update_aer_weight(self, exploration_reward: float) -> float:
    tau = self.config.algorithm.tau  # 目标探索奖励值
    self.aer_weight = self.aer_weight + (tau - exploration_reward)
    self.aer_weight = max(0.0, min(1.0, self.aer_weight))
    return self.aer_weight
```

**计算公式**:

$$w_{t+1} = \text{clip}(w_t + (\tau - \bar{r}_{exp, t}), 0, 1)$$

其中:
- $w_t$ 是第 $t$ 步的权重
- $\tau$ 是目标探索奖励水平（超参数）
- $\bar{r}_{exp, t}$ 是当前batch的平均探索奖励

**直观理解**:
- 如果当前探索奖励 < 目标值 → 增加权重 → 鼓励更多探索
- 如果当前探索奖励 > 目标值 → 减少权重 → 侧重准确性

### 5. 最终奖励

**代码位置**: `aer_ray_trainer.py:149-161`

```python
# 计算当前batch的平均奖励
acc_reward = reward_tensor_acc.sum(-1).mean().item()
exploration_reward = reward_tensor_exploration.sum(-1).mean().item()

# 更新权重
weight = self._update_aer_weight(exploration_reward)

# 组合奖励
combined_reward = reward_tensor_acc + weight * reward_tensor_exploration
batch.batch["rm_scores"] = combined_reward
```

**计算公式**:

$$r_{final}(i) = r_{acc}(i) + w_t \cdot r_{exp}(i)$$

---

## 完整算法伪代码

```
输入: prompt, ground_truth, tau (目标探索奖励), n (每个prompt的响应数)

# 1. 生成阶段
for each prompt:
    for i = 1 to n:
        response[i] = model.generate(prompt)
        uid = unique_id_for_this_prompt

# 2. 奖励计算阶段
for each response[i]:
    # 准确性奖励
    r_acc[i] = 1.0 if verify(response[i], ground_truth) else 0.0

    # 探索奖励
    sim_sum[i] = 0
    for each response[j] in same group:
        sim_sum[i] += token_similarity(response[i], response[j])
    r_exp[i] = 1.0 / sim_sum[i] if sim_sum[i] > 0 else 0.0

# 3. 动态权重更新
avg_exp_reward = mean(r_exp)
weight = clip(weight + (tau - avg_exp_reward), 0, 1)

# 4. 最终奖励
for each response[i]:
    r_final[i] = r_acc[i] + weight * r_exp[i]

return r_final
```

---

## 训练与验证的区别

### 训练模式

**代码**: `aer_ray_trainer.py:149-166`

- 使用 `AERRewardManager`
- 计算准确性 + 探索奖励
- 动态更新权重
- 最终奖励 = 准确性 + 权重 × 探索

### 验证模式

**代码**: `aer_ray_trainer.py:167-169`

- 使用 `RLRewardManager`
- 只计算准确性奖励
- 不使用探索奖励
- 用于评估模型真实性能

---

## 超参数说明

| 参数 | 位置 | 说明 |
|------|------|------|
| `rollout.n` | `ppo_trainer.yaml` | 每个prompt生成的响应数，影响多样性计算 |
| `algorithm.tau` | `ppo_trainer.yaml` | 目标探索奖励值，控制探索强度 |
| `initial_weight` | `aer_ray_trainer.py:93` | 初始权重，默认为0.0 |

---

## 文件结构对应

```
verl/recipe/aer/src/
├── main_ppo.py              # 入口，创建trainer
│   └── 初始化 train_reward_fn (AERRewardManager)
│   └── 初始化 val_reward_fn (RLRewardManager)
│
├── reward_manager.py        # 奖励计算
│   ├── math_accuracy_reward()         # 准确性奖励
│   ├── compute_token_similarity()     # Token相似度
│   ├── RLRewardManager               # 验证用奖励管理器
│   └── AERRewardManager              # 训练用奖励管理器 (AER核心)
│
└── aer_ray_trainer.py       # AER训练器
    ├── __init__()                     # 初始化，提取reward函数
    ├── _compute_aer_reward()          # 计算AER奖励
    ├── _update_aer_weight()           # 更新动态权重
    ├── _compute_reward_colocate()     # 覆盖父类奖励计算
    └── _validate()                    # 验证逻辑
```

---

## 关键设计决策

### 为什么使用token相似度？

Token级别的相似度比最终答案相似度更细粒度：
- 可以区分"正确但推理路径相同"和"正确但推理路径不同"
- 鼓励模型学习不同的解题思路

### 为什么使用倒数 (1/sim)？

- 相似度越高 → 倒数越小 → 惩罚重复
- 相似度越低 → 倒数越大 → 鼓励多样性

### 为什么需要动态权重？

- 训练初期：模型可能缺乏多样性，需要高探索权重
- 训练后期：模型已学会多样化，可以降低探索权重
- 自适应调整避免手动调参



针对AER算法中探索奖励计算时相似度计算方法过于简陋的问题，需要对以下文件进行改进：@verl/recipe/aer/src/reward_manager.py 和 @verl/recipe/aer/src/aer_ray_trainer.py。请设计并实现多种token-level和sequence-level的相似度计算算法，使其能够作为可配置选项在@verl/recipe/aer/run.sh脚本中进行选择。具体要求如下： 1. Token-level相似度算法（至少实现3种）：   - 实现基于余弦相似度的token嵌入比较方法   - 实现基于编辑距离（Levenshtein距离）的token序列比较方法   - 实现基于Jaccard相似度的token集合比较方法   - 每种算法需包含参数配置选项和计算效率优化 2. Sequence-level相似度算法（至少实现3种）：   - 实现基于BERT等预训练语言模型的句子嵌入相似度计算   - 实现基于最长公共子序列（LCS）的序列相似度计算   - 实现基于动态时间规整（DTW）的序列对齐相似度计算   - 每种算法需包含参数配置选项和计算效率优化 3. 配置与集成要求：   - 在@verl/recipe/aer/run.sh中添加相似度算法选择参数（如--similarity-algorithm和--similarity-level）   - 确保算法选择参数能够正确传递到reward_manager.py中的相似度计算模块   - 实现算法注册机制，便于后续扩展新的相似度计算方法   - 添加算法性能基准测试，记录不同算法的计算耗时和内存占用 4. 代码实现要求：   - 代码需符合项目现有的代码风格和注释规范   - 为完成该任务，修改及删除代码时应将原代码注释而不是删去，并用中文注明"# fix: 【详细说明】"；添加代码时，用中文注明"# fix: 【详细说明】" 5. 文档要求：   - 为每种相似度算法提供详细说明，包括原理、适用场景和参数说明   - 更新run.sh脚本的使用文档，说明如何选择和配置不同的相似度算法   - 提供算法性能对比报告，帮助用户根据实际需求选择合适的算法   - 给出现有的相似度计算方式的解析
