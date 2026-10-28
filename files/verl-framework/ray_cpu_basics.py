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
