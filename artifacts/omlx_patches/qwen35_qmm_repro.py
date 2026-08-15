"""Minimal repro: oMLX's qwen35 prefill qmm kernel is slower than MLX's default.

Run with oMLX's bundled interpreter:

  O=/Applications/oMLX.app/Contents/Resources
  PYTHONPATH="$O:$O/Python/framework-mlx-base/lib/python3.11/site-packages" \
    "$O/Python/cpython-3.11/bin/python3.11" qwen35_qmm_repro.py

Observed on M5 Max / oMLX 0.5.8.dev3 (build 260811023551), Qwen3.8-27B MLP
shapes at seq_len=4096:

  gate/up gs128:  mlx 23.5ms   kernel(variant 8) 58.2ms   0.40x
  down    gs128:  mlx 19.0ms   kernel(variant 8) 62.2ms   0.31x
  gate/up gs64 :  mlx 22.1ms   kernel(variant 8) 22.5ms   0.98x  (variant 0: 1.30x)
  down    gs64 :  mlx 20.1ms   kernel(variant 8) 22.8ms   0.88x  (variant 1: 1.13x)

group_size=128 passes every check in _is_supported_affine_linear_shape yet runs
~3x slower than the path it replaces; the patch engages by default above
OMLX_QWEN35_Q4_MLP_MIN_TOKENS=2048, so a 4-bit gs128 model loses ~2x prefill
above 2k context. Measured end-to-end on Qwen3.8-27B-AWQ (4-bit gs128 MLP):

  ppTPS @4k   508.5 -> 917.9   (1.81x)   TTFT  8055ms -> 4462ms
  ppTPS @8k   416.3 -> 822.0   (1.97x)
  ppTPS @16k  375.2 -> 782.9   (2.09x)   TTFT 43663ms -> 20927ms

...simply by setting OMLX_QWEN35_Q4_MLP_MIN_TOKENS high enough to disable it.
"""
import time

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.qwen35_prefill import fast


def bench(fn, n=5):
    mx.eval(fn())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t) / n * 1000


def main():
    mx.random.seed(0)
    qmm = fast.qwen35_q4_affine_qmm_t
    for tag, IN, OUT in (("gate/up 5120->17408", 5120, 17408),
                         ("down    17408->5120", 17408, 5120)):
        x = mx.random.normal((1, 4096, IN)).astype(mx.bfloat16)
        W = mx.random.normal((OUT, IN)).astype(mx.bfloat16)
        for gs in (64, 128):
            ql = nn.QuantizedLinear(IN, OUT, bias=False, group_size=gs, bits=4)
            wq, sc, bi = mx.quantize(W, bits=4, group_size=gs)
            ql.weight, ql.scales, ql.biases = wq, sc, bi
            mx.eval(ql.parameters())
            base = bench(lambda: ql(x))
            for var in (0, 1, 2, 4, 8):
                t = bench(lambda v=var: qmm(x, wq, sc, bi, v, gs))
                flag = " <- oMLX default variant" if var == 8 else ""
                print(f"{tag}  gs{gs:<4d} variant={var}: {t:7.1f} ms  "
                      f"{base/t:5.2f}x vs mlx {base:6.1f} ms{flag}")


if __name__ == "__main__":
    main()
