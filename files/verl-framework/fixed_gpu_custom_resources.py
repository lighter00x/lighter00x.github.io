"""Explicit per-GPU scheduling using application-managed custom resources.

Requires the node startup configuration described in the companion document.
This is a standalone Ray example, not a patch for default verl workers.
"""

import json
import os
import sys

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


TARGETS = [
    ("10.0.0.10", 0),
    ("10.0.0.10", 2),
    ("10.0.0.11", 1),
]


def resource_key(ip, gpu_id):
    return f"gpu_{ip.replace('.', '_')}_{gpu_id}"


@ray.remote
class Worker:
    def __init__(self, gpu_id, key):
        # Ray's normal GPU allocator is disabled for these nodes. The custom
        # resource key provides admission control; this code selects the device.
        # Do not initialize CUDA in a worker setup hook before this constructor.
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            raise RuntimeError("CUDA initialized before explicit device selection")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import torch

        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"GPU {gpu_id} is not accessible as a single device")
        torch.cuda.set_device(0)
        self.probe = torch.zeros(1, device="cuda:0")
        self.gpu_id = gpu_id
        self.key = key
        # Load your model and move all inputs to cuda:0 here.

    def info(self):
        import torch

        ctx = ray.get_runtime_context()
        properties = torch.cuda.get_device_properties(0)
        return {
            "node_id": ctx.get_node_id(),
            "custom_resource": self.key,
            "requested_physical_gpu": self.gpu_id,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_device": "cuda:0",
            "device_name": properties.name,
            "device_uuid": str(getattr(properties, "uuid", "unavailable")),
            # Empty GPU IDs are expected: this application uses custom resources.
            "ray_accelerator_ids": ctx.get_accelerator_ids(),
        }


def main():
    if len(set(TARGETS)) != len(TARGETS):
        raise ValueError("Duplicate node/GPU entries")

    ray.init(address="auto")
    actors = []
    pg = None
    try:
        nodes_by_ip = {}
        for node in ray.nodes():
            if node["Alive"]:
                nodes_by_ip.setdefault(node["NodeManagerAddress"], []).append(node)

        expected_node_ids = []
        keys = []
        for ip, gpu_id in TARGETS:
            matches = nodes_by_ip.get(ip, [])
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one live Ray node for {ip}")
            node = matches[0]
            resources = node["Resources"]
            key = resource_key(ip, gpu_id)
            if resources.get("GPU", 0) != 0:
                raise ValueError(f"{ip} must register --num-gpus=0 for this scheme")
            if resources.get(key, 0) != 1:
                raise ValueError(f"{ip} must register exactly one unit of {key}")
            holders = [
                n for n in ray.nodes()
                if n["Alive"] and n["Resources"].get(key, 0) > 0
            ]
            if len(holders) != 1:
                raise ValueError(f"Custom resource {key} must exist on one node only")
            keys.append(key)
            expected_node_ids.append(node["NodeID"])

        pg = placement_group(
            bundles=[{"CPU": 1, key: 1} for key in keys],
            strategy="PACK",
        )
        ray.get(pg.ready(), timeout=60)

        for index, ((ip, gpu_id), key) in enumerate(zip(TARGETS, keys)):
            actor = Worker.options(
                num_cpus=1,
                num_gpus=0,
                resources={key: 1},
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=index,
                ),
            ).remote(gpu_id=gpu_id, key=key)
            actors.append(actor)

        results = ray.get([actor.info.remote() for actor in actors], timeout=120)
        for result, expected in zip(results, expected_node_ids):
            if result["node_id"] != expected:
                raise RuntimeError(f"Unexpected Actor node: {result}")
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        # Keep actors alive during real training; clean them up at job completion.
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if pg is not None:
            remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
