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
