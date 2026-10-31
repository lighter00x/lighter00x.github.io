---
title: verl 如何调用 Ray：从资源池到一次真实训练
date: '2026-10-31'
permalink: /posts/verl-framework-03/
excerpt: 把 verl 上层设计与 Ray 实际使用串联起来，追踪资源池、Placement Group、WorkerGroup、数据分发与一次训练调用跨越的进程和执行边界。
topic: verl-framework
topic_title: verl 框架使用
topic_title_en: Using the verl framework
math: true
tags:
- verl
- Ray
- 源码阅读
- 分布式训练
---

{% include toc title="文章目录" %}

> 阅读定位：把[verl 上层设计](/posts/verl-framework-01/)与[Ray 实际使用](/posts/verl-framework-02/)接起来。本文同时标明代码运行在哪个进程、使用什么对象、跨过哪条边界。
>
> 2026-10-07 按本地固定快照核对：verl `8718ca30a3f002f93b7c4fd99b9b2506718681bc`，Ray `f54b8b3ea54ba3515f6e8be40258c83b4e00d099`。主线为 V1 sync / CUDA / WorkerDict / 统一 Engine / TransferQueue。V0 和直接传 TensorDict 的教学路径单独标明；不宣称为在线最新 main。

## 1. 先给出真正的两层边界

![图1：从 verl 配置到 Ray 进程，再到模型就绪](/images/verl-framework/07-bridge-startup.png)

verl 决定算法角色、资源布局、训练 rank、batch 分发方式和模型执行目标；Ray 接收资源与远程调用请求，安排进程、执行方法、管理其对象引用。进入远端业务方法以后，Engine 再使用 PyTorch/NCCL 执行模型协作。

所以这不是一条“verl → Ray → NCCL”的继承关系，而是三种责任在一次请求中配合：

| 层 | 负责什么 | 主要入口 |
|---|---|---|
| verl 编排 | 选择角色、分片样本、确定调用顺序 | Trainer、ResourcePool、WG、register |
| Ray 运行时 | 放置 Actors、发送调用、管理依赖和结果 | placement_group、options、remote、get |
| 模型后端 | 分片模型、GPU 计算和集合通信 | Engine、DeviceMesh、torch.distributed |

## 2. 外层 Driver 怎样把控制权交给 TaskRunner

执行位置：启动脚本的 Python 进程。入口是 `main_ppo.py`：

```python
# 源码关系精简，省略配置转换和 runtime_env 合并。
ray.init(**ray_init_kwargs)
runner = task_runner_class.remote()
ray.get(runner.run.remote(config))
```

这里 `runner` 是 ActorHandle，`runner.run.remote(config)` 是一次远程方法调用；外层 Driver 等待结果。V1 的 `TaskRunnerV1.run()` 在自己的进程中初始化 TQ、创建 Trainer、初始化管理器并进入训练：

```python
# 本地 V1 主关系；省略异常清理和 manager 构造细节。
tq.init(config.transfer_queue)
self.trainer = trainer_cls(config=config)
self.trainer.init()
self.init_agent_loop_manager()
self.trainer.fit(self.agent_loop_manager)
```

此时 Trainer 是 TaskRunner 进程内的普通对象。TaskRunner 虽然是 Ray worker 进程，也能继续提交 Actors 和远程任务。下文的“控制端”指这个组织训练的进程，不总是最外层 Driver。

## 3. 三种 mapping：职责、规模和落点规划

执行位置：控制端。V1 的 `_init_resource_pool_mgr()` / `_setup()` 组织角色与资源。

```python
# 教学描述，角色实际按算法条件选择。
role_worker_mapping = {
    Role.ActorRolloutRef: ray.remote(ActorRolloutRefWorker),
}

resource_pool_spec = {"global_pool": [2, 2]}

mapping = {
    Role.ActorRolloutRef: "global_pool",
}
```

第一张表是 Role→Worker 类；第二张表是池名→计划布局；第三张表是 Role→池名。`ray.remote(Class)` 只包装类，尚未创建训练实例。

本文设两台各 2 张空闲 GPU 的节点；每组 2 个槽位，共 4 rank，`world_size=sum([2,2])=4`。这是布局规格，不是 IP 清单。下面使用同构布局，因为本地普通路径以 `store[0]` 组织 local_world_size 和遍历；不能从此推断任意 `[2,4]` 布局已被完整支持。

ResourcePoolManager 创建 RayResourcePool 对象并检查资源总量，但真正的预留要继续调用 PG 接口。总量检查通过仍可能因为节点容量或 CPU 约束导致 PG pending。

## 4. ResourcePool 怎样变成 Ray Placement Groups

执行位置：控制端发起，Ray 运行时处理资源放置。本地 `RayResourcePool.get_placement_groups()` 的普通 CUDA 关系是：

```python
# 源码逻辑精简。m 表示当前实际 max_colocate_count。
bundle = {"CPU": m, "GPU": 1}
pg_scheme = [
    [bundle.copy() for _ in range(count)]
    for count in process_on_nodes
]
pgs = [
    placement_group(bundles=bundles, strategy="STRICT_PACK")
    for bundles in pg_scheme
]
ray.get([pg.ready() for pg in pgs])
```

`[2,2]` 变成两个 PG，每个两个单 GPU bundles。STRICT_PACK 要求一个 PG 内部同节点；在恰有两台各 2 卡容量时两组分别落地。若有 4 卡节点，两组仍可能同节点。

此时确定资源放置，但还没有加载模型或建立训练通信。PG 与 Actor 的额度关系是“预留一组资源，再从里面使用”，不是两次向集群重复扣账。

本地会在就绪后查询 PG 的实际节点 IP 并排序，主要帮助 rank 顺序稳定。**排序发生在放置之后，不会把 PG 调度到指定 IP，更不保证固定物理 GPU。**

## 5. RayClassWithInitArgs：从构造说明到 remote

执行位置：控制端。构造说明保存远程类和参数：

```python
actor_cls = RayClassWithInitArgs(
    cls=ray.remote(ActorRolloutRefWorker),
    config=config.actor_rollout_ref,
    role="actor_rollout_ref",
)
```

这不是训练 Actor 实例。真正创建时，`RayWorkerGroup` 选择 PG/bundle，准备 rank 环境，再调用这个封装。普通创建分支最终执行：

```python
# 本地 RayClassWithInitArgs.__call__ 的精简关系。
options = {
    "scheduling_strategy": PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=bundle_index,
    ),
}
options.update(saved_options)
options["num_gpus"] = num_gpus  # CUDA 平台上的示意展开
handle = self.cls.options(**options).remote(*self.args, **self.kwargs)
```

`.options()` 保存创建选项；`.remote()` 提交创建并返回 ActorHandle。Ray 根据已选 bundle、资源需求和运行环境启动/准备 Actor，执行其构造逻辑；handle 返回不证明构造完成。

## 6. RayWorkerGroup 怎样给每个 rank 注入身份

执行位置：控制端。普通 CUDA 路径先通过首个 PG 的辅助任务选 master 地址/端口，再为各槽位确定全局 rank。`_create_worker()` 组织：

```python
# 本地关键字段；省略名称、profiler 与平台分支。
num_gpus = 1 / resource_pool.max_colocate_count
env_vars = {
    "WORLD_SIZE": str(world_size),
    "RANK": str(rank),
    "WG_PREFIX": self.name_prefix,
    "WG_BACKEND": "ray",
    "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
    "MASTER_ADDR": self._master_addr,
    "MASTER_PORT": self._master_port,
}
ray_cls_with_init.update_options({
    "runtime_env": {"env_vars": env_vars},
    "name": actor_name,
})
```

master 是训练 rendezvous 入口，不必等于 Ray head。Ray 负责准备环境，verl 负责选择这些训练字段的内容。

| 排序后的 PG | bundle index | RANK | WORLD_SIZE | RAY_LOCAL_WORLD_SIZE |
|---|---:|---:|---:|---:|
| PG 0 | 0 | 0 | 4 | 2 |
| PG 0 | 1 | 1 | 4 | 2 |
| PG 1 | 0 | 2 | 4 | 2 |
| PG 1 | 1 | 3 | 4 | 2 |

本地这个环境字典没有把 bundle 槽位直接注入为 `LOCAL_RANK`。在普通单卡可见路径中 Worker 的 LOCAL_RANK 默认可为 0，因此每个进程都能使用自己的 `cuda:0`。

假设 rank 3 的 bundle 对应节点 B 上物理 GPU 0：`bundle_index=1`、`RANK=3`、`CUDA_VISIBLE_DEVICES=0`、进程内设备 `cuda:0`。这四者分别描述资源槽位、通信身份、可见列表和 CUDA 索引，不能互相推导。

`m=3` 时 Actor 请求 `1/3 GPU`，但 bundle 预留 1 GPU。它允许额外进程共享额度，不表示每个模型最多用 1/3 显存，也不自动创建三份角色。CPU 额度还需看 Actor 自己是否显式声明 `num_cpus`，不能把 PG 的 `CPU:m` 当成所有默认 Actor 的运行期 CPU 事实。

## 7. 实际 V1 多了一层 WorkerDict 组合

执行位置：控制端组合类，远端构造普通对象。下面是**需要完整 config 和已连接 Ray 的接口示例**，不是独立训练脚本：

```python
import ray
from verl.single_controller.ray import (
    RayResourcePool, RayClassWithInitArgs, RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.workers.engine_workers import ActorRolloutRefWorker

pool = RayResourcePool(
    process_on_nodes=[2, 2], use_gpu=True,
    max_colocate_count=1, name_prefix="bridge",
)
class_dict = {
    "actor_rollout_ref": RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role="actor_rollout_ref",
    ),
}
container_cls = create_colocated_worker_cls(class_dict)
physical_wg = RayWorkerGroup(
    resource_pool=pool, ray_cls_with_init=container_cls,
)
views = physical_wg.spawn(prefix_set=class_dict.keys())
actor_wg = views["actor_rollout_ref"]
actor_wg.init_model()
```

这里只有第一次 WorkerGroup 创建时启动四个外层 Actors。`spawn` 复用 handles，再绑定角色前缀方法，不新增四个 GPU 进程。

远端 WorkerDict 的关键构造关系是：

```python
# 精简自本地源码；不是新的用户 Worker 实现。
class WorkerDict(worker_cls):
    def __init__(self):
        super().__init__()
        self.worker_dict = {}
        for key, remote_cls in cls_dict.items():
            user_cls = _unwrap_ray_remote(remote_cls)
            with temp_env_var("DISABLE_WORKER_INIT", "1"):
                self.worker_dict[key] = user_cls(
                    *init_args_dict[key]["args"],
                    **init_args_dict[key]["kwargs"],
                )
```

注意内部调用是 `user_cls(...)`，不是 `.remote()`。业务入口被绑定到 WorkerDict 上，因此角色视图的方法能映射到 `actor_rollout_ref_compute_log_prob` 等带前缀入口；部分 fused 路径则通过统一 execute 方法转发。

## 8. 从外层容器进入 TrainingWorker 和 Engine

执行位置：每个远端训练进程。启用 ref 的合并角色中，`init_model()` 先构造 ref，后构造 actor；下面是精简骨架：

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    self.ref = self.ref_worker_cls(config=ref_training_config)
    self.ref.reset()
    self.set_dispatch_collect("ref", **self.ref.get_dispatch_collect())

    self.actor = self.actor_worker_cls(config=actor_training_config)
    self.actor.reset()
    self.actor.set_loss_fn(self.loss_fn)
    self.set_dispatch_collect("actor", **self.actor.get_dispatch_collect())
    # 随后准备 rollout mesh、适配组件和权重同步组件。
```

真实源码按 role 和配置执行各分支；未启用 ref 时不创建它。LoRA 参考路径还可能通过 actor 禁用 adapter 完成。

内部 TrainingWorker 的构造是：

```python
# 精简自 TrainingWorker.__init__。
Worker.__init__(self)
initialize_global_process_group_ray(timeout_second=None)
self.engine = EngineRegistry.new(
    model_type=config.model_type,
    backend=config.engine_config.strategy,
    model_config=config.model_config,
    engine_config=config.engine_config,
    optimizer_config=config.optimizer_config,
    checkpoint_config=config.checkpoint_config,
)
self._register_dispatch_collect_info(
    mesh_name="train",
    dp_rank=self.engine.get_data_parallel_rank(),
    is_collect=self.engine.is_mp_src_rank_with_outputs(),
)
```

`reset()` 再调用 `engine.initialize()`。global process-group helper 检查是否已初始化；同进程多个 TrainingWorker 不会无条件重复建 WORLD。本地 NVIDIA helper 可选择 `cpu:gloo,cuda:nccl`，不是所有平台一律只有 NCCL。`timeout=None` 使用后端默认语义，不等于无限等待。

初始化请求必须先向所有参与 rank 提交，再统一等待。若等 rank 0 的 collective 初始化结束才向 rank 1 发请求，前者可能一直等不到后者。

## 9. Engine 拓扑怎样变成 WG 分发信息

![图2：Engine 拓扑投影为 dispatch 与 collect](/images/verl-framework/08-bridge-dispatch.png)

这里有三种不同含义：

1. **DeviceMesh / Megatron parallel state**：真实模型通信与分片拓扑。
2. **内部 `"train"`**：TrainingWorker 保存自己的 `dp_rank/is_collect` 的 key。
3. **外部 `"actor"/"ref"`**：外层业务对象按角色暴露的信息 key。

外层 `set_dispatch_collect` 复制/登记 metadata，不创建 NCCL group。方法上的 `@register` 选择已有 metadata，也不创建 mesh。

```python
@register(
    dispatch_mode=make_nd_compute_dataproto_dispatch_fn("actor")
)
def compute_log_prob(self, data):
    output = self.actor.infer_batch(data)
    return output.cpu() if output is not None else None
```

装饰器决定“按谁的分发拓扑发数据”；函数体决定“让哪个模型计算”。这两个维度由代码保持一致，装饰器不会自动纠正函数体调用错误。

`make_nd_compute_dataproto_dispatch_fn` 返回一对函数：

```python
return {
    "dispatch_fn": partial(dispatch_lazy_compute_data_proto, mesh_name),
    "collect_fn": partial(collect_lazy_compute_data_proto, mesh_name),
}
```

控制端首次使用某个 mesh 时查询全部成员信息并缓存。dispatch 获得 `dp_rank_mapping`，collect 获得布尔 mask。

### 两种不能混用的具体例子

| 配置示例 | dispatch mapping | collect mask | 32 条轨迹的含义 |
|---|---|---|---|
| 4 rank 纯 FSDP，无 SP | `[0,1,2,3]` | `[T,T,T,T]` | 等量时每 rank 8 条不同轨迹 |
| 4 rank、DP=2、TP=2 的教学布局 | `[0,0,1,1]` | `[T,F,T,F]` | 每 DP shard 16 条，每组两 rank 协作 |

纯 FSDP 分片的是模型状态，dispatch 仍可让各 rank 处理不同数据；collect=True 不意味着该 rank 永久保存完整模型。TP 布局中，TP rank 的输出可能重复或仅指定来源有效，所以按 backend 信息收集。

本地纯 FSDP 实现，在没有 Ulysses mesh 时，`get_data_parallel_rank()` 返回 global rank，DP size 返回 WORLD size，输出标记为 True。开启 SP 后会变化。复杂 Megatron 映射应查实际 parallel state，不能只凭整除公式推断。

## 10. 一次 WG 调用怎样变成 Ray 方法调用

执行位置：控制端。方法代理是在 WorkerGroup 绑定注册信息时生成的。`func_generator()` 核心关系：

```python
# 精简自 Ray 适配层，省略 padding 清理。
def proxy(*args, **kwargs):
    args, kwargs = dispatch_fn(worker_group, *args, **kwargs)
    output = execute_fn(method_name, *args, **kwargs)
    if blocking:
        output = ray.get(output)
    return collect_fn(worker_group, output)
```

`execute_fn` 进一步选中 Actor handles 和远程入口，通常提交 `remote_call.remote(...)`。可见 WG 的普通方法外观包住了 dispatch、Ray submit、等待和 collect。

`blocking` 是方法注册选项。并不是所有 WG 方法都直接返回最终 tensor/list：非阻塞路径可能返回引用、future 或封装对象，由后续控制逻辑完成等待。不能把所有调用都机械展开成同一种 `ray.get(list)`。

对于 TP 的教学 mapping，分发逻辑可以理解为：

```python
# 可独立验证映射的纯 Python 示例，不是 verl dispatch 的完整实现。
data = list(range(32))
dp_chunks = [data[:16], data[16:]]
mapping = [0, 0, 1, 1]
worker_inputs = [dp_chunks[dp_rank] for dp_rank in mapping]
assert worker_inputs[0] == worker_inputs[1]
assert worker_inputs[2] == worker_inputs[3]

mask = [True, False, True, False]
worker_outputs = [list(chunk) for chunk in worker_inputs]
kept = [out for out, collect in zip(worker_outputs, mask) if collect]
merged = [value for out in kept for value in out]
assert merged == data
```

真实代码处理 DataProto、TensorDict、metadata、padding 和 future 等，而不是只处理 list。collect 选择输出来源，不表示其他 ranks 可以不执行模型计算，也不表示框架一定从未为其他 ranks 创建结果引用。

## 11. Ray 接到请求后：创建与稳定调用分别追踪

### 11.1 创建阶段

`.options(...).remote(...)` 将 Actor 类、构造参数、资源需求和环境交给 Ray。底层涉及 CoreWorker 的 Actor 创建、注册和 GCS Actor 调度，目标 raylet 分配/获取 worker，然后执行外层构造。

代码入口可以沿：

```text
verl RayClassWithInitArgs.__call__
→ Ray Python/Cython API 边界
→ CoreWorker::CreateActor
→ Actor 创建/注册与 GCS 协作
→ GcsActorScheduler::Schedule / CreateActorOnWorker
→ 节点 worker 执行构造逻辑
```

这个图是逻辑职责链，省略异步回调、重试、租约和运行环境准备，不是每个版本固定的逐函数同步栈。

### 11.2 方法调用阶段

Actor 已存在时，一次 `method.remote()` 通过 CoreWorker 提交 Actor task，处理依赖和目标连接，ActorTaskSubmitter 发送请求。目标 CoreWorker 接收并进入 Python 执行逻辑。

```text
WG execute / remote_call.remote
→ CoreWorker::SubmitActorTask
→ ActorTaskSubmitter 处理提交、依赖和队列
→ PushActorTask
→ 目标 CoreWorker 接收任务
→ 远端业务方法
→ 返回值完成 / Ray 对象处理
```

已有地址后的稳定调用可以直达 Actor worker；不会每次再由 GCS 选一个任意 GPU 或新建模型。GCS 的生命周期管理与每次方法数据传输要分开理解。

## 12. V1 的完整调用：Ray 控制、TQ 数据、NCCL 计算

![图3：一次 compute_log_prob 跨过三条线路](/images/verl-framework/09-bridge-call.png)

以 `actor_rollout_wg.compute_log_prob(batch_meta)` 为例：

| 顺序 | 执行位置 | 发生什么 |
|---|---|---|
| 1 | 控制端 Trainer | 准备需要计算的样本 keys/字段与执行信息 |
| 2 | 控制端 WG | 查询/缓存 actor mapping，分发 meta |
| 3 | 调用方 Ray 运行时 | 提交 Actor 方法，生成结果引用 |
| 4 | 目标 worker | 接收调用，register 的 TQ bridge 处理输入 |
| 5 | TQ bridge | 获取真实字段，构成 TensorDict |
| 6 | 业务 Worker | `compute_log_prob → self.actor.infer_batch` |
| 7 | TrainingWorker / Engine | eval/no_grad 路径执行模型前向 |
| 8 | GPU 后端 | 按 FSDP/TP 等规则做计算和通信 |
| 9 | 远端 bridge | 按 collect 约定写回输出字段、返回 metadata/指标 |
| 10 | 控制端 | 等待并收集结果，进入后续训练阶段 |

本地 bridge 的相关函数包括 `_async_meta_to_realdata()`、`_async_update_meta_with_output()` 和收集判定。没有必要把 TQ 的所有真实字段都画成“先 ray.put，再 Plasma 传给每个 rank”。TQ 有独立存储实现；Ray 仍处理远程调用参数和返回对象。

若使用直接传 TensorDict/DataProto 的路径，调用参数本身承载更多真实数据，Ray 会按相应对象的序列化与传输路径处理。**直接数据路径与 V1 meta/TQ 路径可共用业务计算接口，但数据位置不同。**

`update_actor` 的后半段换成：

```text
Worker.update_actor
→ TrainingWorker.train_mini_batch：epoch / mini-batch
→ TrainingWorker.train_batch
→ Engine.train_batch
→ zero_grad
→ forward_backward_batch：micro-batch，模型前向、loss、backward
→ optimizer_step
→ 返回训练指标
```

这里没有每轮重新创建 Actor，也没有 Ray 自动替用户建立 TP 或平均梯度。Ray 送调用；Engine 使用已经建立的通信组完成模型协作。

## 13. 生成调用为什么不能简单画成 actor WG generate

V1 中生成主要沿 AgentLoop / LLM server 路径组织，而非把训练 WorkerGroup 上的 actor mesh 直接当成生成入口：

```text
TaskRunner 初始化 AgentLoopManager
→ Trainer 的生成/ReplayBuffer 流程组织请求
→ AgentLoopWorker 接收并组织任务
→ llm_client 请求服务
→ LLMServerManager 管理的 rollout replica
→ vLLM/SGLang 执行生成
→ 轨迹字段写入 TQ，完成状态供 ReplayBuffer 采样
```

rollout mesh、服务副本和请求 Actor 数量各有定义。普通无 prefill/decode 分离的布局中，副本 GPU 需求受 TP×rollout DP×PP 影响；hybrid replica 可以关联已有训练 handles/资源，但仍可能启动额外服务进程。不能据“资源复用”断定“生成全在同一个 Python PID”。

旧 V0 的 `async_rollout_manager.generate_sequences(...)` 等调用可帮助理解，但本篇 V1 主线必须沿 `fit`、生成提交与 ReplayBuffer 的实际实现阅读。

## 14. 训练完成后还有权重调用和显存交接

本地 V1 sync 在采样结束 sleep replicas，在 step 结束调用 `CheckpointEngineManager.update_weights()`。管理器再组织训练端导出、传输和推理端加载。

`ActorRolloutRefWorker.update_weights()` 是 `async def`，注册 `blocking=False`。naive 共置路径从 actor Engine 导出参数并更新 rollout 适配组件；其他 checkpoint backends 可以走发送/接收路径。模型布局转换与传输是具体 backend 的职责。

部分非 naive 同步路径还会用现有 replica handles 临时构造一个 RayWorkerGroup。这里包装已有 handles，不代表新增一套训练 GPU Actors。发送侧和接收侧需要先匹配提交，再等传输完成。

一次闭环中，三类同步目的分别是：

- WG/Ray 完成等待：这个阶段的远程方法是否结束。
- Engine collective：参与同一模型计算的 ranks 是否完成匹配通信。
- train→rollout 权重同步：下一批生成是否使用更新后的参数。

这些等待不能互相替代。`ray.get` 一个 rank 的返回值不证明所有其他 rank 都完成了；FSDP 梯度同步完成也不证明推理权重已更新。

## 15. GPU 定制接入时，为什么不能只改一个选项

原资料的每卡自定义资源方案是独立 Ray 示例，默认 verl 不直接适配。若需要 rank→节点→GPU UUID 的固定映射，至少要保持以下环节一致：

| 环节 | 必须同步考虑的变化 |
|---|---|
| ResourcePoolManager | 从通用 GPU 总量检查改为目标每卡额度检查 |
| PG bundles | 根据映射声明对应 custom resource，而不是统一 GPU:1 |
| Actor options | 选择 custom resource，避免又申请通用 GPU |
| Worker 初始化 | 在 CUDA 初始化前设置实际设备选择 |
| rank/PG 排序 | 用 PG 身份和 bundle 身份追踪设备，避免排序错位 |
| Engine | rank 集合、device 和并行规模一致 |
| rollout/reward/teacher | 所有相关 GPU 使用遵循同一套额度 |

自定义资源名只是调度额度；Ray 不按 `gpu_xxx_2` 字符串自动选择物理 GPU 2。独立例子采用节点 `--num-gpus=0`、Actor `num_gpus=0`，应用自行绑定。容器设备权限和 UUID 仍需验证。

如果需求只是限制某几张卡，先在节点服务启动前限定可见集合会更直接；如果只是稳定 checkpoint rank 顺序，先核对节点排序。固定 UUID 会限制故障迁移，需接受设备不可用时的等待或失败。

## 16. 用 rank 3 检查你是否读通链路

在本文两机四 rank 的纯 FSDP 设定中，rank 3 的完整经历是：

```text
控制端规划 [2,2]
→ 两个 PG 落地、排序
→ PG 1 / bundle 1 被安排为 RANK=3
→ runtime_env 注入 WORLD_SIZE=4 与共同 master
→ Ray 准备外层 WorkerDict Actor 和 GPU 可见性
→ 内部 ActorRolloutRefWorker 普通对象构造
→ init_model 创建内部 TrainingWorker
→ rank 3 加入 WORLD，构造并初始化 Engine
→ 纯 FSDP 的 dispatch identity 为 3，collect=True
→ 每次收到它的 DP 数据份额 / metadata
→ TQ 取数据，Engine 做前向/训练与集合通信
→ 写回输出并返回指标
→ 更新后的策略权重送到 rollout
→ 同一进程继续下一轮
```

rank 身份通常不每轮重分；变的是数据、权重、版本和显存驻留。物理设备绑定是否固定，要看实际调度和生命周期，不能靠 rank 编号保证。

## 17. 本地源码索引：按创建链和调用链阅读

| 入口 | 重点符号 |
|---|---|
| [main_ppo.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/trainer/main_ppo.py) | `run_ppo`、`TaskRunnerV1.run` |
| [trainer_base.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/trainer/ppo/v1/trainer_base.py) | `_init_resource_pool_mgr`、`_setup`、`_step_once` |
| [ray/base.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/single_controller/ray/base.py) | PG、`RayClassWithInitArgs.__call__`、`_create_worker`、`spawn`、`func_generator` |
| [base/worker_group.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/single_controller/base/worker_group.py) | `world_size`、注册方法绑定 |
| [base/worker.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/single_controller/base/worker.py) | 环境读取、dispatch/collect 信息保存 |
| [base/decorator.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/single_controller/base/decorator.py) | `register`、dispatch、collect |
| [engine_workers.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/workers/engine_workers.py) | `init_model`、TrainingWorker、业务入口 |
| [distributed.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/utils/distributed.py) | `initialize_global_process_group_ray` |
| [transferqueue_utils.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/utils/transferqueue_utils.py) | meta→TensorDict→输出字段 |
| [engine/base.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/workers/engine/base.py) | `EngineRegistry`、`train_batch`、`infer_batch` |
| [checkpoint_engine/base.py](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/checkpoint_engine/base.py) | `CheckpointEngineManager.update_weights` |
| [Ray core_worker.cc](https://github.com/ray-project/ray/blob/f54b8b3ea54ba3515f6e8be40258c83b4e00d099/src/ray/core_worker/core_worker.cc) | Actor 创建、方法提交和任务接收 |
| [Ray gcs_actor_scheduler.cc](https://github.com/ray-project/ray/blob/f54b8b3ea54ba3515f6e8be40258c83b4e00d099/src/ray/gcs/actor/gcs_actor_scheduler.cc) | Actor 创建放置 |
| [Ray actor_task_submitter.cc](https://github.com/ray-project/ray/blob/f54b8b3ea54ba3515f6e8be40258c83b4e00d099/src/ray/core_worker/task_submission/actor_task_submitter.cc) | 发送到已有 Actor 的请求 |

读每个位置都记录四件事：**当前在哪个进程；手里是普通对象、ActorHandle 还是 ObjectRef；真实数据在哪；本次等待由谁负责。**这样就能把上层设计、下层 API 和实际运行时放进同一条证据链。

## 18. 原文整合与验证范围

原目录六份 Markdown、资源定制 HTML 和配套示例均作为资料保留。新三篇分别抽取设计、API 使用和源码衔接，避免把不同入口的代码拼成一个伪“完整脚本”。图中的布局均注明教学条件。

本地源码快照可复查；本次未在真实集群完成 GPU、DDP、FSDP 或完整 GRPO 运行。第二篇的完整脚本有清楚前提，第一、第三篇的精简骨架用于解释源码，不属于可直接部署的训练作业。静态验证结果见配图目录中的 `validation.json`。
