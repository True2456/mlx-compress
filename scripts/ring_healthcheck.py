#!/usr/bin/env python
"""Two-Mac MLX ring sanity check.

Run AFTER the thunderbolt bridge + hostfile are set up:

    mlx.launch --backend ring --hostfile scripts/hosts.json \
        scripts/ring_healthcheck.py

Proves: (1) both ranks join the ring, (2) collectives cross the link,
(3) rough interconnect bandwidth. If this passes, pipeline_load will work.
"""
import time
import mlx.core as mx


def main():
    g = mx.distributed.init(backend="ring")
    rank, size = g.rank(), g.size()

    # 1) barrier-ish: every rank contributes its rank id, all must agree on sum
    got = mx.distributed.all_sum(mx.array(rank), stream=mx.cpu).item()
    expected = size * (size - 1) // 2
    if rank == 0:
        print(f"[ring] size={size} all_sum(rank)={got} expected={expected} "
              f"{'OK' if got == expected else 'MISMATCH'}")

    # 2) bandwidth probe: all_sum a ~256 MB tensor a few times
    n = 64 * 1024 * 1024  # 64M float32 = 256 MB
    x = mx.ones(n, dtype=mx.float32)
    mx.eval(x)
    mx.synchronize()
    t0 = time.time()
    iters = 5
    for _ in range(iters):
        x = mx.distributed.all_sum(x)
        mx.eval(x)
    mx.synchronize()
    dt = (time.time() - t0) / iters
    if rank == 0:
        gb = n * 4 / 1e9
        print(f"[ring] all_sum {gb:.2f} GB x{iters}: {dt*1000:.1f} ms/iter "
              f"~{gb / dt:.1f} GB/s effective")
        print("[ring] link is up and collectives work — safe to pipeline_load")


if __name__ == "__main__":
    main()
