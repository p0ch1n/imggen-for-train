"""Shared CLI helpers."""
import argparse
import time

from .config import get, load_config


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=True, help="pipeline yaml")
    return p


def add_shard_args(p):
    p.add_argument("--shard", type=int, default=0, help="this worker's shard index (multi-GPU: one process per GPU)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="process at most N items (quick trial runs)")


def shard(items, args):
    items = [x for i, x in enumerate(items) if i % args.num_shards == args.shard]
    return items[: args.limit] if args.limit else items


def load(args):
    return load_config(args.config)


class Progress:
    def __init__(self, total, name, every=20):
        self.total, self.name, self.every = total, name, every
        self.n, self.t0 = 0, time.time()

    def step(self, k=1):
        self.n += k
        if self.n % self.every == 0 or self.n == self.total:
            el = time.time() - self.t0
            eta = el / self.n * (self.total - self.n) if self.n else 0
            print(f"[{self.name}] {self.n}/{self.total}  {el / max(self.n, 1):.2f}s/it  eta {eta / 60:.1f} min",
                  flush=True)


__all__ = ["base_parser", "add_shard_args", "shard", "load", "get", "Progress"]
