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
