---
title: verl 上层设计：从强化学习算法到分布式执行
date: '2026-10-27'
permalink: /posts/verl-framework-01/
excerpt: 从算法角色、Trainer、WorkerGroup、Worker 与 Engine 的职责出发，梳理一轮强化学习训练中的数据流、权重流和分布式执行关系。
topic: verl-framework
topic_title: verl 框架使用
topic_title_en: Using the verl framework
math: true
tags:
- verl
- 源码阅读
- 分布式训练
---

{% include toc title="文章目录" %}

> 阅读定位：先理解 verl 为什么拆成这些组件，以及它们如何共同完成一轮训练。Ray 的使用细节见[第二篇：Ray 实际使用](/posts/verl-framework-02/)，两层如何接起来见第三篇（verl 调用 Ray，待发布）。
>
> 本文围绕组件职责、训练闭环和分布式执行展开，依据固定源码快照整理。原稿源码核对日期：2026-10-07；verl 快照：`8718ca30a3f002f93b7c4fd99b9b2506718681bc`。主线为 V1 sync、CUDA、统一 Engine 路径；V0、异步、LoRA 和独立 rollout 在相应位置单独说明。示例中的数字是教学设定。

## 1. verl 的核心任务：让数据流与权重流形成闭环

普通监督学习可以直接拿输入和标签训练。语言模型强化学习多出了一步：模型要先生成回答，再评价这些回答，然后用评价改变生成概率。因此，一个训练系统至少需要协调生成、打分、概率计算、参数更新和权重同步。

![图1：verl 的分层职责](/images/verl-framework/01-verl-design.png)

这里始终有两条流：

- **轨迹数据流**：prompt → response → reward → log probability → advantage → loss。
- **策略权重流**：训练参数 θk → 推理端使用 θk 生成 → 训练得到 θk+1 → 推理端接收 θk+1。

一轮训练只有在新参数送到推理端之后，才对下一轮采样产生作用。`optimizer.step()` 完成仅表示训练端已经更新。

本文使用一个贯穿例子：8 个 prompt，每个采样 4 个 response，共 32 条轨迹；训练用 4 个 GPU rank；策略采用 FSDP，不额外启用 TP/SP/CP；rollout 的 TP=1；普通 GRPO 不创建 Critic；按需启用固定参考模型。模型必须能满足单卡推理和共置显存要求。这不是可直接运行的完整配置。

## 2. 先分清“算法角色”和“运行实体”

| 名称 | 回答的问题 | 典型内容 |
|---|---|---|
| RL Actor / Policy | 训练哪个策略？ | 正在被更新的语言模型 |
| Rollout | 谁生成回答？ | vLLM/SGLang 推理后端及适配组件 |
| Reference | 用什么约束策略偏移？ | 固定参考模型，或 LoRA 禁用 adapter 的路径 |
| Critic | 怎样估计状态价值？ | value model，常用于 PPO/GAE |
| Reward | 怎样判断回答质量？ | 规则函数、奖励模型、外部评价 |
| Ray Actor | 哪个远程进程持有状态？ | 控制器、训练 Worker、服务 Actor 等 |
| WorkerGroup | 怎样调用一组执行者？ | 维护 handles，分发参数并收集结果 |

**一个角色不必对应一个独立进程，一个进程也可以包含多个角色。**这使算法设计能先表达职责，再由部署层选择共置或分离。

普通 GRPO 通常利用同一 prompt 的多条回答建立相对 baseline，因此不需要训练 Critic。Reference 是否启用是另一个选择，主要取决于 KL 的使用方式。不能从“没有 Critic”推出“没有 Reference”。

## 3. 为什么存在 Trainer、WorkerGroup、Worker、Engine 四层

### 3.1 Trainer：规定一轮训练的依赖关系

Trainer 负责选择算法、组织样本、调用各阶段、记录指标、保存和恢复状态。它决定“先算什么、再更新什么”。

在该快照的 V1 入口中，外层 Python Driver 启动远程 `TaskRunnerV1`，后者创建普通 Trainer 对象并运行训练循环。因此，要区分启动程序进程和真正运行 Trainer 的控制器进程。

### 3.2 WorkerGroup：把一次调用展开为一组调用

Trainer 希望写：

```python
# 接口示意：batch_meta 是已准备好的 V1 样本元数据。
output_meta = actor_rollout_wg.compute_log_prob(batch_meta)
```

WorkerGroup 根据方法注册信息，决定数据按几份切、哪些 rank 收到同一份、哪些输出需要收集，再执行 Ray 远程调用。它是控制端封装，不是 GPU 上的模型。

### 3.3 Worker：暴露角色业务入口

`ActorRolloutRefWorker` 的入口可以转到内部不同组件：

```python
# 本地调用关系的精简骨架；省略装饰器、CPU 转换和可选分支。
def compute_log_prob(self, data):
    return self.actor.infer_batch(data)

def compute_ref_log_prob(self, data):
    return self.ref.infer_batch(data)

def update_actor(self, data):
    return self.actor.train_mini_batch(data)
```

`TrainingWorker` 提供通用训练/前向接口，管理 mini-batch、epoch、loss 函数和指标。它既能用于训练策略，也能只做参考概率或 value 前向。

### 3.4 Engine：落实模型计算和状态管理

Engine 持有模型、优化器、学习率调度器及相关分布式拓扑，执行 forward、backward、参数更新、offload 和 checkpoint。具体后端通过统一接口接入：FSDP、Megatron、VeOmni、TorchTitan 等实现各自的计算方式。

这套分层的收益是：算法主要通过稳定的 Worker 接口表达意图，而并行计算细节集中在 Engine 中。换训练后端不意味着重写每一处 Trainer 调用。

## 4. 共置设计：把多个逻辑对象装进同一进程

![图2：控制端视图与远端对象树](/images/verl-framework/02-verl-objects.png)

该快照的 V1 按资源池组合业务类，可以生成外层 `WorkerDict` Ray Actor。它在进程内直接实例化各角色对象。一个 rank 的典型对象树是：

```text
WorkerDict：外层 Ray Actor，拥有一个 PID
└─ worker_dict["actor_rollout_ref"]
   └─ ActorRolloutRefWorker：进程内业务对象
      ├─ self.actor → TrainingWorker → 策略 Engine
      ├─ self.ref   → TrainingWorker → 参考 Engine（按需）
      └─ self.rollout → 推理适配组件
```

如果直接将 `ActorRolloutRefWorker` 用 `.remote()` 实例化，它就是外层 Ray Actor；如果它被 `WorkerDict` 普通构造，它只是内部对象。**判断进程边界要看创建调用，而不能只看类名。**

实际推理服务还可能启动额外 Ray Actors 和后端子进程；上面的图只表达训练容器的对象关系。

`spawn()` 为已有 Actor handles 建立角色访问视图。例如 actor WG 和 critic WG 可以指向同一批外层 Actors。新增视图不会凭空增加一套 GPU 进程，也不能把各 WG 的 `world_size` 简单相加计算卡数。

另一种共置是不同进程共享 GPU 逻辑额度，例如各申请 `num_gpus=0.5`。这种共享与同进程对象组合不同：前者有多个 PID，后者有多个对象。两种方式都需要另外管理显存。

## 5. 初始化设计：先有进程，再有模型和分发信息

关键顺序是：

```text
角色/资源规划
→ PG 预留资源
→ Ray 创建外层 Actors，注入 rank 和 master 信息
→ 外层业务对象构造，actor/ref/rollout 先为空
→ WG 广播 init_model()
→ 创建内部 TrainingWorker
→ 确保进程级 torch.distributed WORLD 已初始化
→ EngineRegistry.new(...) 选择并构造 Engine
→ reset() → engine.initialize() 加载模型等状态
→ 将 Engine 的 dp_rank / collect 标记登记到外层
→ 创建推理服务管理器，恢复 checkpoint，首次同步权重
```

本地合并角色路径中，启用 Reference 时会先创建 ref，再创建 actor；同进程第二次执行 global process-group helper 会检查已有初始化状态。actor/ref 的模型对象和业务状态仍各自独立。

`init_model()` 使用 `Dispatch.ONE_TO_ALL`：初始化前还没有 `"actor"/"ref"` 分发拓扑，必须先把初始化任务广播给全部 rank，才能得到这些信息。这是启动依赖，不能让初始化反过来依赖尚未建立的 mesh。

Engine 的构造和初始化也要分开：部分拓扑会在 Engine 构造阶段建立，而模型、优化器等在 `initialize()` 完成。具体时机以对应后端为准。

## 6. 数据抽象设计：样本字段与控制元数据分开

早期路径常直接传 `DataProto`，其中组织 tensor 和非 tensor 字段；内部模型接口可以使用 `TensorDict`。V1 主线采用 TransferQueue，把真实样本字段存储与控制调用分开。

| 对象 | 所在层 | 作用 |
|---|---|---|
| DataProto | verl 数据层 | 组织一批样本与元信息 |
| TensorDict | 模型调用层 | 给 Worker/Engine 提供 tensor 字段 |
| KVBatchMeta / BatchMeta | V1 控制与存储桥接 | 指定样本 keys、partition、字段和执行信息 |
| ObjectRef | Ray 层 | 表示某个远程结果或 Ray 对象 |
| CUDA tensor | GPU 计算层 | 真正参与模型计算的张量 |

典型 V1 调用是：

```text
Trainer 传 meta
→ WG 按 Engine 提供的 DP 信息分发 meta
→ Ray 送到目标进程
→ tqbridge 从 TQ 取得真实字段，构成 TensorDict
→ Worker/Engine 计算
→ 输出字段按约定写回 TQ
→ 控制端获得输出 metadata 和指标
```

该快照默认 TQ 配置描述的是 `SimpleStorage` / ZMQ 内存存储。它不是 Ray Plasma 的另一个名字。Ray 负责调用和其参数/返回对象，TQ 负责自身的数据存取；控制器在算 advantage 等阶段也可能读真实字段。

## 7. 一轮 GRPO：从 32 条轨迹到更新

![图3：一轮同步训练中的数据和权重变化](/images/verl-framework/03-verl-loop.png)

### 7.1 先采样，再确定可训练样本

AgentLoop 组织生成请求和多轮交互，LLMServerManager 管理推理服务副本，后端执行 token 采样。Reward 可以随轨迹完成，也可以在训练前补算。ReplayBuffer 依据完成状态、过滤和版本条件选择样本。

教学设定是 8×4=32 条，但过滤、补齐、partial rollout 和采样策略可能改变实际数量。`rollout.agent.num_workers` 是请求处理 Actor 的数量，不等于训练 GPU rank 数量。

### 7.2 补齐 old/ref/value，再算 advantage

该快照的 V1 `_step_once()` 的主顺序为：采样 → 按需补奖励 → balance → old logprob → 按需 ref logprob → 按需 values → advantage/return → 按需 critic 更新 → actor 更新。

| 概率 | 回答的问题 | 更新期间的状态 |
|---|---|---|
| rollout logprob | 轨迹实际由哪个行为策略采出？ | 记录采样时的信息 |
| old logprob | 当前策略相对本批固定 anchor 改变多少？ | 本批更新期间固定 |
| current logprob | 现在的训练参数给这些 token 多大概率？ | 每次训练前向重新计算 |
| ref logprob | 策略偏离参考模型多少？ | 参考策略通常固定 |

默认重算 old 与复用 rollout 的 bypass 路径不同。异步采样或后端数值约定不同会让 rollout 与 old 不一致，要沿配置检查 correction，不能默认替换。

### 7.3 一个可独立运行的 GRPO 数值例子

下面仅计算组内优势，需 PyTorch；不包含模型更新：

```python
import torch

rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
mean = rewards.mean()
std = rewards.std()  # 本地实现默认使用样本标准差
group_adv = (rewards - mean) / (std + 1e-6)

response_mask = torch.tensor([
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [1, 0, 0, 0],
], dtype=torch.float32)
token_adv = group_adv[:, None] * response_mask
print(group_adv.tolist())   # 约 [0.866, -0.866, 0.866, -0.866]
print(token_adv)
```

这些回答按 prompt UID 分组，不按 GPU rank 分组。正优势推动回答概率上升，负优势推动下降；实际单次变化还受 clipping、KL 和优化器等影响。所有奖励相同时，这个组内信号为零，但其他损失项仍可能产生梯度。

### 7.4 策略 loss 怎样使用这些字段

经典 clipped surrogate 使用概率比：

$$r_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t)).$$

下面是已经对齐到 response token 的张量上的**教学损失函数**，不是 verl 全部 loss 的替代实现：

```python
def clipped_policy_loss(current_logp, old_logp, advantages, mask, eps=0.2):
    ratio = (current_logp - old_logp).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantages
    per_token = -torch.minimum(unclipped, clipped)
    return (per_token * mask).sum() / mask.sum().clamp_min(1)
```

真实实现还处理跨 rank/global token 归一化、loss aggregation、KL、entropy、不同 policy loss 和 rollout correction。模型 logits 与目标 token 的位置必须正确错位对齐，padding 不能作为有效 response。

## 8. Engine 执行设计：mini-batch 与 micro-batch

`TrainingWorker.train_mini_batch()` 管 epoch 和 mini-batch 迭代；`BaseEngine.train_batch()` 的核心是：

```python
# 精简自本地 BaseEngine，不是独立程序。
self.optimizer_zero_grad()
outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
grad_norm = self.optimizer_step()
```

以 32 条轨迹、全局 mini-batch=16、epoch=2、4 个 FSDP 数据分发 rank 为例：每 epoch 两个 mini-batch，合计四次 optimizer step；每 rank 每 mini-batch 处理 4 条轨迹。若 micro-batch=2，每 rank 两次前向/反向后更新一次。这里的 mini-batch 以最终轨迹数计；原始 `ppo_mini_batch_size` 在入口可能以 prompt 数计并乘 `rollout.n`。

FSDP 虽然分片模型状态，各 rank 仍可处理不同数据。在本地不启用 Ulysses SP 的 FSDP 路径中，dispatch DP rank 是 global rank，collect 标记都为 True。**“模型分片”不能推导成“只有 rank 0 有有效样本输出”。**

TP 示例则不同：4 rank、DP=2、TP=2 时，映射可以是 `[0,0,1,1]`，两个 TP rank 处理同一 DP shard，仅收集每组规定的输出来源。这个教学布局适用于相应模型并行后端，不能直接套到上述纯 FSDP 设定。

梯度通信由 Engine 后端执行。DDP 常见梯度 AllReduce；全分片 FSDP 常见参数 AllGather 与梯度 ReduceScatter；TP/PP 有各自协作。同步时机随梯度累积和后端策略变化，Ray 不负责把梯度收回控制器再平均。

## 9. 权重同步和显存生命周期

训练端和推理端可能有不同权重布局、名称、进程和缓存。更新后需要导出、转换并传给 rollout。`CheckpointEngineManager` 在这里主要组织权重传输；这与面向恢复的磁盘 checkpoint 保存不是同一个操作。

V1 sync 的关键 hooks 是：

```python
# 本地 trainer_sync.py 的核心关系，省略计时和 profiling。
def on_init_end(self):
    self.checkpoint_manager.update_weights(self.global_steps)

def on_sample_end(self):
    self.checkpoint_manager.sleep_replicas()

def on_step_end(self):
    self.checkpoint_manager.update_weights(self.global_steps)
```

首次同步发生在 checkpoint 恢复之后。采样完成后推理 sleep，为训练腾出显存；训练结束后同步、恢复推理。具体释放权重、KV cache 或 offload 什么，由后端与配置决定。

进程可以持续存活，而 GPU 上的张量驻留随阶段变化。Actor 仍占 Ray 逻辑资源时，即使显存暂时空闲，也不表示卡已归还给其他作业。Ray object spilling 管对象存储，不能自动卸载训练 activation 或推理 KV cache。

## 10. 同步和异步改变了哪些设计约束

| 模式 | 主要时间关系 | 需要额外关注 |
|---|---|---|
| sync | 采样、训练、同步按轮衔接 | 权重交接完成，显存切换 |
| colocate_async | 请求/轨迹组织可异步，共置 GPU 仍需阶段协调 | pause/abort、sleep、resume 的边界 |
| separate_async | 独立推理 GPU 可生成，训练 GPU 同时消费 | 样本版本、staleness、校正与同步 |

Actor 方法声明为 `async def` 不表示训练和生成 kernel 自动同时运行，也不表示样本没有旧策略问题。`separate_async` 中是否还启用 hybrid replicas，要看配置；名称本身不是完整部署图。

故障恢复也分层：Ray 重启 Actor 不会自动恢复 optimizer、调度器、dataloader、TQ、ReplayBuffer 和策略版本。参数更新有副作用，重试不能直接当作 exactly-once 训练。

## 11. 源码阅读路线与自查

| 阅读入口（固定版本源码） | 带着什么问题读 |
|---|---|
| [main_ppo.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/trainer/main_ppo.py) | Driver 怎样启动 TaskRunnerV1？ |
| [V1 trainer_base.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/trainer/ppo/v1/trainer_base.py) | `_setup`、`fit`、`_step_once` 怎样组织依赖？ |
| [engine_workers.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/workers/engine_workers.py) | 角色入口怎么调用 TrainingWorker？ |
| [engine/base.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/workers/engine/base.py) | 统一 Engine 接口规定什么？ |
| [FSDP 实现](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/workers/engine/fsdp/transformer_impl.py) | micro-batch、模型前向和参数更新在哪里？ |
| [transferqueue_utils.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/utils/transferqueue_utils.py) | meta 怎么变成真实数据并写回？ |
| [trainer_sync.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/trainer/ppo/v1/trainer_sync.py) | sleep 和权重同步何时发生？ |

读懂本篇后，应能回答：一个角色为什么不等于一个 PID？为什么 old 要固定？Reference 和 Critic 各服务什么目的？为什么 optimizer step 后还要同步？哪些行为属于 Ray，哪些属于 Engine？