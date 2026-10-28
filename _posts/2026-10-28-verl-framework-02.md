---
title: Ray 实际使用：从远程调用到 GPU 联合训练
date: '2026-10-28'
permalink: /posts/verl-framework-02/
excerpt: 从可运行的小程序出发，理解 Ray 的远程调用、Actor 状态、对象传递、GPU 与 Placement Group 调度，以及 Ray 和 PyTorch
  DDP 的分工。
topic: verl-framework
topic_title: verl 框架使用
topic_title_en: Using the verl framework
math: true
tags:
- Ray
- 分布式训练
- GPU
---

{% include toc title="文章目录" %}

> 阅读定位：从能运行的小程序出发，理解 Ray 如何创建进程、保存状态、传递数据、分配 GPU，并与 PyTorch 通信配合。本文不依赖 verl；与 verl 的对应关系见第三篇（verl 调用 Ray，待发布）。
>
> 依据原目录资料及本地 Ray 快照 `f54b8b3ea54ba3515f6e8be40258c83b4e00d099`，整理于 2026-10-07。Python API 需与实际安装版本核对；CPU 示例不要求 GPU，NCCL 示例面向 Linux CUDA 环境。本次做静态核对与语法验证，未在 Ray/GPU 集群执行。

三个完整示例已另存为脚本，与正文代码一致：[CPU 基础](/files/verl-framework/ray_cpu_basics.py)、[GPU/PG 探针](/files/verl-framework/ray_pg_probe.py)、[两 GPU DDP](/files/verl-framework/ray_ddp_two_gpu.py)。可以先读对应章节的环境前提，再在合适环境运行脚本。

## 1. 三条线路：调用、数据、状态

![图1：Ray 运行时组件及其职责](/images/verl-framework/04-ray-runtime.png)

Ray 程序同时有三条线：控制线提交和执行任务，数据线传递参数与结果，状态线维护 Actor、对象和集群信息。

| 名称 | 实体 | 主要作用 |
|---|---|---|
| Driver | 启动用户主程序的进程 | 提交计算、组织结果 |
| Head node | 集群节点 | 承载 GCS，也可执行计算 |
| Worker node | 集群计算节点 | 提供本机资源与执行进程 |
| raylet | 每节点守护进程 | 资源调度、worker 管理与节点级对象工作 |
| Ray worker | 用户代码执行进程 | 执行 Task 或承载 Actor |
| CoreWorker | 嵌在 Driver/worker 中的运行时 | 提交、接收任务，管理引用和 RPC |
| GCS | 集群控制服务 | 节点、Actor 等集群状态管理 |
| Object Store | 每节点共享对象存储 | 保存和共享部分参数与结果 |
| Object owner | 首次创建该引用的进程 | 对象元数据与引用生命周期管理 |

Worker node 是节点，Ray worker 是进程。CoreWorker 不是“又一台工作节点”。GCS 不充当所有训练数据、参数、结果的中央转发站。

## 2. 先启动或连接 Ray

### 2.1 本地 CPU 练习

在安装好 Ray 的隔离 Python 环境中：

```python
import ray

ray.init(num_cpus=4)
try:
    print(ray.cluster_resources())
    print(ray.available_resources())
finally:
    ray.shutdown()
```

`ray.init()` 初始化连接，在常见无现成实例的本机场景中启动本地运行时。Driver 就是正在运行此 Python 程序的进程，不是 `init()` 另行创建的主程序。

### 2.2 两机或多机集群

以下是 Linux shell 示例，地址按实际环境替换；每个节点应有兼容的 Ray/Python、代码和依赖：

```bash
# head：此例让控制节点不提供 GPU
ray start --head --port=6379 --num-cpus=4 --num-gpus=0

# GPU 节点：在该机器上执行
ray start --address=10.0.0.10:6379 --num-cpus=16 --num-gpus=4
```

程序连接已有集群：

```python
ray.init(address="auto")
```

找不到可连接实例时，这个连接请求会失败。`ray.shutdown()` 对远程集群主要是断开当前程序，不会把所有远端节点服务关闭。若本地运行时由该程序启动，shutdown 会清理相应本地运行时。

`cluster_resources()` 是集群登记的逻辑总量；`available_resources()` 是异步更新的可调度视图。它们都不是 GPU 剩余显存表。

## 3. Task、Actor 和普通对象怎样选

| 写法 | 创建/提交的东西 | 状态是否适合跨调用保留 |
|---|---|---|
| `f(x)` | 当前进程函数调用 | 当前进程普通状态 |
| `RemoteFunction.remote(x)` | 一次 Ray Task | 不应依赖调用间共享进程状态 |
| `RemoteClass.remote()` | 一个 Ray Actor | 适合长期持有模型、优化器、缓存 |
| `handle.method.remote(x)` | 对已有 Actor 的一次方法调用 | 访问 Actor 已有实例状态 |
| `Class()` | 当前进程普通对象 | 没有因此产生 Ray 进程 |

`@ray.remote` 包装定义，本身还没有执行计算。Actor 创建返回 `ActorHandle`；Task 和通常的 Actor 方法调用返回结果 `ObjectRef`。两类引用用途不同。

## 4. 完整 CPU 示例：看见进程、状态和依赖

保存为 `ray_cpu_basics.py`。运行前确认可用 CPU 逻辑资源不少于 4：

```python
import os
import socket
import ray


@ray.remote(num_cpus=1)
def make_numbers(n):
    return list(range(n))


@ray.remote(num_cpus=1)
def square(x):
    return x * x


@ray.remote(num_cpus=1)
class Accumulator:
    def __init__(self):
        self.calls = 0
        self.total = 0

    def consume(self, numbers):
        # numbers 是实际 list；顶层 ObjectRef 已被运行时解析。
        self.calls += 1
        self.total += sum(numbers)
        return {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "calls": self.calls,
            "total": self.total,
        }


def main():
    ray.init(num_cpus=4)
    actor = None
    try:
        print("driver pid:", os.getpid())
        actor = Accumulator.remote()  # ActorHandle，构造可能尚未完成
        data_ref = make_numbers.remote(10)  # ObjectRef
        first = actor.consume.remote(data_ref)
        second = actor.consume.remote(data_ref)
        print(ray.get([first, second], timeout=60))

        # 先提交全部，再等全部。ray.get(list) 按输入引用顺序返回。
        refs = [square.remote(i) for i in range(6)]
        print(ray.get(refs, timeout=60))

        # 不在 Driver 上提前 get 中间结果。
        a = square.remote(3)
        b = square.remote(a)
        print(ray.get(b, timeout=60))
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
```

预期观察：Actor 两次输出 PID 相同，calls=1、2，total=45、90；平方列表是 `[0,1,4,9,16,25]`；依赖链最后得到 81。PID 的具体值不固定。这个单机实验展示进程与引用关系，不证明跨节点传输。

普通同步、单线程 Actor 在同一提交者的正常调用序列中按相应顺序执行；不能把这个结论扩大为多个提交者、异步 Actor、并发配置、重试下的全局顺序保证。

## 5. ObjectRef：引用与数据位置分别跟踪

![图2：对象引用、owner 和跨节点数据](/images/verl-framework/05-ray-objects.png)

```python
input_ref = ray.put(large_data)
result_ref = actor.consume.remote(input_ref)
```

`ray.put` 在调用进程所在节点创建对象存储数据。它不会立即广播到所有节点。远端需要输入时，由运行时按需取得副本。

假设 Driver 在 A、Actor 在 B：input_ref 的 owner 是 Driver，初始数据在 A；result_ref 通常也由调用方法的 Driver 创建并拥有，但大结果最初由 B 上的执行进程产生。**owner 所在位置与数据所在位置可以不同。**

| 情况 | 常见默认数据路径 |
|---|---|
| 普通小参数/结果 | 可序列化后内联在 RPC 中 |
| 普通大参数 | 可隐式存入本地对象存储，以引用传递 |
| `ray.put(data)` | 显式存储，适合多次复用同一大输入 |
| 顶层 ObjectRef | 作为依赖解析，远端收到实际值 |
| list/dict 内的 ObjectRef | 嵌套引用保留，代码按需取得值 |

小/大对象的具体阈值取决于实现与配置，不必背固定数值。共享内存的零拷贝也只适用于相应对象类型和条件，不能扩展成跨节点、任意 Python 对象或 CPU→GPU 全链路零拷贝。

下面展示顶层与嵌套引用，需在已初始化 Ray 的程序中执行：

```python
@ray.remote(num_cpus=1)
def inspect_top(value):
    return type(value).__name__

@ray.remote(num_cpus=1)
def inspect_nested(values):
    value = ray.get(values[0])
    return type(values[0]).__name__, type(value).__name__

ref = ray.put([1, 2, 3])
print(ray.get(inspect_top.remote(ref)))       # list
print(ray.get(inspect_nested.remote([ref])))  # ObjectRef, list
```

对象引用生命周期也重要：`del ref` 仅删除一个 Python 引用；容器和远程调用可能仍持有它。owner 死亡与某个对象副本丢失是不同故障，不能假设任何对象都能无限重建。

## 6. get、wait 与并发提交

`ray.get` 返回实际值；`ray.wait` 返回已就绪/未就绪引用。后者可帮助按完成顺序消费任务：

```python
# 放在前面的已初始化程序内；square 已定义。
pending = [square.remote(i) for i in range(12)]
while pending:
    ready, pending = ray.wait(pending, num_returns=1, timeout=5)
    if ready:
        print(ray.get(ready[0]))
```

`ray.wait` 默认可能把对象拉到本地；`fetch_local=False` 用于只关心远端就绪等场景，不能一概说 wait 不传数据。

下面两个写法的并发机会不同：

```python
# 每次提交后立即等待，循环很容易串行。
values = [ray.get(square.remote(i)) for i in range(8)]

# 先把任务都交给运行时，再统一等待。
refs = [square.remote(i) for i in range(8)]
values = ray.get(refs)
```

先提交再等只是提供并发机会；真正并发还取决于资源、依赖和执行者并发方式。大量任务应限制在途数量，避免排队和对象内存增长。

异步 Actor 的协程需要可让出的等待点；阻塞计算会阻塞其事件循环。线程并发 Actor 又是另一套机制。`.remote()` 异步返回引用不代表用户函数变成异步函数。

## 7. GPU：Ray 负责资源准入，PyTorch 负责计算

| 设置 | 含义 | 不能推出什么 |
|---|---|---|
| `num_gpus=1` | 请求一个逻辑 GPU，默认设置可见设备 | 自动加载模型或限制显存 |
| `num_gpus=0.5` | 允许共享 GPU 调度容量 | 每进程只能用一半显存 |
| `num_gpus=2` | 一个 Actor 请求同节点两张 GPU | 自动把模型做 TP |
| `model.to("cuda")` | 将模型放进进程可用 GPU | 自动建立跨进程训练通信 |

一个 Actor 的 `num_gpus=8` 要求某一节点能满足 8 GPU，不会自动拼接两台各 4 卡机器。跨节点训练通常是多个单 GPU Actor/rank 协作。

GPU Actor 持有资源的时间通常是其生命周期。一次方法结束或模型 offload 不等于 GPU 逻辑额度释放。Task 的额度则服务一次任务；GPU Task 的 worker 回收策略与普通 CPU Task 也可能不同。

单 GPU Actor 中，宿主机 GPU 3 可能通过 `CUDA_VISIBLE_DEVICES=3` 暴露为进程内 `cuda:0`。rank 3 不意味着使用 `cuda:3`。容器、UUID 和 MIG 会使编号关系更复杂，精确检查应记录实际 GPU UUID。

## 8. 完整 GPU/PG 示例：两张卡、两个 Actor

![图3：PG 预留、Actor 消耗与显存是三个账本](/images/verl-framework/06-ray-placement.png)

这个探针不训练模型，要求已有集群、同一节点有 2 GPU 和 2 CPU 可供一个 STRICT_PACK PG 预留。保存为 `ray_pg_probe.py`：

```python
import os
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=1, num_gpus=1)
class GPUProbe:
    def info(self):
        import torch
        torch.cuda.set_device(0)
        x = torch.ones(4, device="cuda:0")
        return {
            "rank": os.environ["RANK"],
            "node": ray.get_runtime_context().get_node_id(),
            "pid": os.getpid(),
            "ids": ray.get_runtime_context().get_accelerator_ids(),
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_count": torch.cuda.device_count(),
            "value": x.sum().item(),
        }


def main():
    ray.init(address="auto")
    pg = None
    actors = []
    try:
        pg = placement_group(
            [{"CPU": 1, "GPU": 1} for _ in range(2)],
            strategy="STRICT_PACK",
        )
        ray.get(pg.ready(), timeout=60)
        for rank in range(2):
            actor = GPUProbe.options(
                runtime_env={"env_vars": {"RANK": str(rank)}},
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=rank,
                ),
            ).remote()
            actors.append(actor)
        print(ray.get([a.info.remote() for a in actors], timeout=120))
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if pg is not None:
            remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
```

读这段代码要区分三个完成点：`pg.ready()` 证明预留与放置就绪；`.remote()` 返回 handle 表示已提交 Actor 创建；`info()` 结果就绪证明远端已完成该探针。PG 并不提前为模型分配 CUDA 显存。

PG bundles 描述预留量；Actor 的资源声明描述从 bundle 使用多少。因此 PG 与 Actor 各写一次 `GPU:1` 不是向节点重复申请两份。指定 PG 而不声明 Actor 的 GPU 需求，也不能默认它消耗该 GPU 额度。

### PG 策略与两机布局

| 策略 | 约束/倾向 |
|---|---|
| PACK | 尽量集中到少量节点 |
| SPREAD | 尽量分散到不同节点 |
| STRICT_PACK | 全部 bundles 必须同一节点 |
| STRICT_SPREAD | 每个 bundle 必须不同节点 |

每个 bundle 本身必须装进单个节点。PG 原子预留以单个 PG 为边界；两个独立 PG 不是一起成功或一起失败的一次事务。

两台各 4 卡机器上可以创建两个各含四个单 GPU bundles 的 STRICT_PACK PG。在恰好只有这些容量的条件下，它们分别占节点；若有一台 8 卡机器，两个 PG 也可能同节点。`[4,4]` 和 PG 数量都不是物理节点 ID 清单。

## 9. 从 GPU Actor 到联合训练：显式建立通信组

上面的两个 Actor 各自计算只是两个独立 GPU 工作。联合训练还需要：统一的 `WORLD_SIZE`、唯一的 `RANK`、共同 rendezvous 地址/端口、匹配的训练调用和并行后端。

下面是**完整的两 GPU DDP 教学程序**，要求 Linux、兼容的 CUDA PyTorch/NCCL/Ray、同一节点至少 2 GPU/2 CPU；环境中没有额外的设备可见性覆盖。初始化放在显式 `init_model()`，便于对应 verl。保存为 `ray_ddp_two_gpu.py`：

```python
import os
import socket
from datetime import timedelta
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=1)
def endpoint():
    # 空闲端口探测是教学方式，释放后有被其他进程占用的竞争窗口。
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    return ray.util.get_node_ip_address(), port


@ray.remote(num_cpus=1, num_gpus=1)
class TrainRank:
    def __init__(self):
        self.rank = int(os.environ["RANK"])

    def init_model(self):
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        if torch.cuda.device_count() != 1:
            raise RuntimeError("本例要求每个 Actor 只看到一张 GPU")
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(seconds=120),
        )
        torch.manual_seed(7)
        self.model = DDP(torch.nn.Linear(4, 1).cuda(), device_ids=[0])
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.01)
        return {"rank": self.rank, "pid": os.getpid()}

    def train_once(self):
        import torch
        import torch.distributed as dist

        torch.manual_seed(100 + self.rank)
        x = torch.randn(8, 4, device="cuda:0")
        target = x.sum(dim=1, keepdim=True)
        self.optim.zero_grad(set_to_none=True)
        loss = ((self.model(x) - target) ** 2).mean()
        loss.backward()  # DDP 在这里使用 NCCL 同步梯度
        self.optim.step()

        marker = torch.tensor(float(self.rank), device="cuda:0")
        dist.all_reduce(marker)
        return {
            "rank": self.rank, "loss": loss.item(),
            "rank_sum": marker.item(),
            "weight_sum": self.model.module.weight.detach().sum().item(),
        }

    def close(self):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    ray.init(address="auto")
    pg = None
    actors = []
    try:
        pg = placement_group(
            [{"CPU": 1, "GPU": 1} for _ in range(2)],
            strategy="STRICT_PACK",
        )
        ray.get(pg.ready(), timeout=60)

        def strategy(index):
            return PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=index,
            )

        addr, port = ray.get(
            endpoint.options(scheduling_strategy=strategy(0)).remote(),
            timeout=60,
        )
        for rank in range(2):
            actor = TrainRank.options(
                scheduling_strategy=strategy(rank),
                runtime_env={"env_vars": {
                    "RANK": str(rank), "WORLD_SIZE": "2",
                    "MASTER_ADDR": addr, "MASTER_PORT": str(port),
                }},
            ).remote()
            actors.append(actor)

        # 全部提交，再等待；不能初始化 rank 0 并等完才初始化 rank 1。
        print(ray.get([a.init_model.remote() for a in actors], timeout=180))
        print(ray.get([a.train_once.remote() for a in actors], timeout=180))
        ray.get([a.close.remote() for a in actors], timeout=60)
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if pg is not None:
            remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
```

预期：rank 集合是 `{0,1}`，两者 rank_sum=1，更新后的 weight_sum 接近一致，局部 loss 可以不同。并非所有指标天然一致；DDP 同步梯度，而各 rank 的输入不同。

多机扩展时，要把 PG 分组、rendezvous 节点选择、唯一 rank、网络可达性和初始化并发一起改对。增加 Actor 数量不等于已经实现多机通信。`MASTER_ADDR` 不必是 Ray head；rendezvous 端口也不是每轮全部梯度必经的中央转发端口。

## 10. 节点和 GPU 定制：控制范围要明确

### 10.1 固定节点

未使用 PG 时，可以用 `NodeAffinitySchedulingStrategy(node_id=..., soft=False)`。它固定节点，不固定该节点上的物理卡。使用 PG 时，Actor 选项指定的是 PG/bundle，节点约束应在 PG 创建阶段安排；不能把两个不同的策略对象同时塞进一个 `scheduling_strategy`。

已有资料提到 PG 的 `bundle_label_selector`。此类 API 有版本条件，使用前检查实际签名：

```python
import inspect
from ray.util.placement_group import placement_group
print(inspect.signature(placement_group))
```

### 10.2 限定节点可用设备集合

在该节点启动 Ray 服务之前限制，例如 Linux：

```bash
CUDA_VISIBLE_DEVICES=0,2 ray start --address=10.0.0.10:6379 --num-gpus=2
```

这是节点部署设置。它不保证 Actor 0 拿物理卡 0、Actor 1 拿卡 2，也不即时修改已有服务。

### 10.3 每个 Actor 必须固定到具体卡

可沿原资料的[自定义每卡资源示例](/files/verl-framework/fixed_gpu_custom_resources.py)继续学习。该方案要求节点不登记通用 GPU 额度，统一登记每卡自定义资源，例如 `gpu_node_a_2:1`；PG 预留这个资源，Actor 使用 `num_gpus=0, resources={key:1}`，应用在 CUDA 初始化前选择对应设备。

资源名不会自动让 Ray 识别“这是物理 GPU 2”。调度记账与真实设备选择的一致性由应用维护，不能与另一套普通 GPU 分配混用。最好验证实际 UUID；普通 `get_accelerator_ids()` 没有 GPU 项在此方案下是预期现象。

默认 verl 还检查通用 GPU 总量，并为 rollout 等组件申请 GPU，所以独立 Ray 示例成功不表示可以直接套进 verl。第三篇列出需同步改造的环节。

## 11. 创建 Actor 与调用方法的底层路径

理解 API 后，再读运行时。Actor 有两个阶段：

```text
创建阶段
Class.remote()
→ CoreWorker 组织 Actor 创建规格
→ Actor 注册/创建与 GCS Actor 调度协作
→ raylet 提供资源和 worker
→ 在目标 worker 中执行 __init__

方法阶段
handle.method.remote(...)
→ CoreWorker 提交 Actor task
→ 处理参数依赖、顺序和目标连接
→ ActorTaskSubmitter 把请求送到目标 Actor worker
→ 执行方法，处理返回对象
→ 调用方取得结果
```

Actor 地址已知后的稳定方法调用不需要每次重新申请 GPU或创建进程。GCS 仍参与生命周期管理，但它不是每次方法 RPC 的固定串行中转点。

普通 Task 的路径不同：提交者可以向 raylet 申请 worker lease，取得 worker 地址后发执行请求；lease 可复用。普通 CPU Task worker 通常可执行后续任务，Task 无状态不等于任务结束就销毁进程。

本地底层入口：

| 入口 | 阅读内容 |
|---|---|
| [CoreWorker](https://github.com/ray-project/ray/blob/f54b8b3ea54ba3515f6e8be40258c83b4e00d099/src/ray/core_worker/core_worker.cc) | `CreateActor`、`SubmitActorTask`、`HandlePushTask` |
| [GcsActorScheduler](https://github.com/ray-project/ray/blob/f54b8b3ea54ba3515f6e8be40258c83b4e00d099/src/ray/gcs/actor/gcs_actor_scheduler.cc) | 创建 Actor 的调度与 worker 分配 |
| [ActorTaskSubmitter](https://github.com/ray-project/ray/blob/f54b8b3ea54ba3515f6e8be40258c83b4e00d099/src/ray/core_worker/task_submission/actor_task_submitter.cc) | `PushActorTask`、连接和结果回调 |
| [TaskManager](https://github.com/ray-project/ray/blob/f54b8b3ea54ba3515f6e8be40258c83b4e00d099/src/ray/core_worker/task_manager.cc) | 返回对象完成与 inline/共享存储处理 |
| [NodeManager](https://github.com/ray-project/ray/blob/f54b8b3ea54ba3515f6e8be40258c83b4e00d099/src/ray/raylet/node_manager.cc) | 节点运行时工作 |

本地 Ray checkout 是 sparse checkout，不能假设所有 Python API 文件均在磁盘。API 使用可参考本地 `doc/source/ray-core/`；以上入口反映固定快照，不是所有 Ray 版本的固定函数位置。

## 12. 排障和释放：按阶段观察

| 现象 | 优先检查 |
|---|---|
| `ray.init` 失败 | 地址、节点、Python/Ray 环境和网络 |
| `pg.ready` 超时 | 每节点资源、单 bundle 容量、严格放置策略、标签 |
| handle 返回但 Actor 不可用 | 构造函数、runtime_env、日志和 Actor 状态 |
| 方法迟迟不开始 | 前一个同步方法、输入依赖、Actor 并发配置 |
| `init_process_group` 超时 | rank 缺失/重复、WORLD_SIZE、master、网络和设备 |
| backward OOM | 模型、activation、micro-batch、累积和后端策略 |
| 函数完成而 get 很慢 | 序列化、对象传输、spilling 和内存压力 |
| owner 死亡错误 | 引用拥有进程及其生命周期 |
| 训练重试后状态异常 | 副作用重复、checkpoint 恢复、幂等性和版本 |

可用环境中，配合 `ray status`、`ray list actors`、`ray list tasks`、Dashboard 和 worker 日志；State CLI 需要相关服务和依赖。多机 NCCL 还需看各 rank 的后端日志。

作业退出时先停止业务 Actor，再删除 PG。Actor 停止主要把额度归还 PG 内；删除 PG 才释放整体预留给集群。`del handle`、`ray.kill`、`remove_placement_group`、`ray.shutdown` 的语义不能互换。Detached Actor/PG 需要另行管理生命周期。

Ray Actor 重启不会自动恢复模型训练状态。任务超时也不自动等于取消远端执行；有副作用的训练方法不能盲目重试。

完成 CPU 示例后，你应能解释引用依赖；完成 PG 探针后，应能区分可见设备与 rank；读完 DDP 示例，应能区分 Ray 的方法调用与 NCCL 的梯度通信。接下来读第三篇（verl 调用 Ray，待发布），看 verl 将这些操作封装到了哪里。
