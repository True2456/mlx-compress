# Qwen3.5 prefill qmm kernel is ~4x slower than the MLX default at group_size=128

## Summary

`omlx/patches/qwen35_q4_mlp.py` replaces `Qwen3_5MLP`'s quantized matmuls with
`qwen35_q4_affine_qmm_t` for prompts longer than `OMLX_QWEN35_Q4_MLP_MIN_TOKENS`
(default 2048). `_is_supported_affine_linear_shape` accepts `group_size` of
either 64 or 128, but at 128 the kernel runs about 4x slower than the
`nn.QuantizedLinear.__call__` path it replaces.

4-bit / group_size 128 is the default layout produced by `mlx_lm.convert` and
`mlx-community` quants, so any Qwen3.5 or 3.6 model in that shape loses roughly
half its prefill throughput above 2k context, silently, with no log line to
indicate a slow path was taken.

At group_size 64 the kernel is exactly at parity with the default (1.00x), so
on this hardware it does not appear to be earning its keep in either
configuration.

## Environment

- oMLX 0.6.0.dev1, build 260814000805-macos26-27
- MLX 0.32.0
- Apple M5 Max, 128 GB, macOS 26.4 (25E246)
- Model: Qwen3.8-27B quantized to 4-bit group_size 128 on the MLP
  (attention 8-bit gs64, GatedDeltaNet in_proj 5-bit gs64, lm_head 6-bit gs128)

## Reproduction

Standalone, uses only the shipped kernel and MLX:

```python
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

mx.random.seed(0)
qmm = fast.qwen35_q4_affine_qmm_t
for tag, IN, OUT in (("gate/up", 5120, 17408), ("down   ", 17408, 5120)):
    x = mx.random.normal((1, 4096, IN)).astype(mx.bfloat16)
    W = mx.random.normal((OUT, IN)).astype(mx.bfloat16)
    for gs in (64, 128):
        ql = nn.QuantizedLinear(IN, OUT, bias=False, group_size=gs, bits=4)
        wq, sc, bi = mx.quantize(W, bits=4, group_size=gs)
        ql.weight, ql.scales, ql.biases = wq, sc, bi
        mx.eval(ql.parameters())
        base = bench(lambda: ql(x))
        t8 = bench(lambda: qmm(x, wq, sc, bi, 8, gs))
        print(f"{tag} gs{gs:<4d} mlx {base:6.1f} ms  kernel(v8) {t8:6.1f} ms  {base/t8:.2f}x")
```

Run it with the bundled interpreter:

```
O=/Applications/oMLX.app/Contents/Resources
PYTHONPATH="$O:$O/Python/framework-mlx-base/lib/python3.11/site-packages" \
  "$O/Python/cpython-3.11/bin/python3.11" repro.py
```

Result, sequence length 4096, variant 8 (the shipped default):

```
gate/up gs64   mlx  13.0 ms   kernel  12.9 ms   1.00x
gate/up gs128  mlx  12.8 ms   kernel  50.7 ms   0.25x
down    gs64   mlx  14.0 ms   kernel  13.9 ms   1.00x
down    gs128  mlx  13.8 ms   kernel  51.3 ms   0.27x
```

Every variant 0 through 8 was tried at gs128. All land between 0.21x and
0.26x, with the shipped default (8) the best of a bad set at 3.8x slower than
the MLX path:

```
gate/up gs128, mlx default 13.1 ms
  variant 0: 54.0 ms   variant 3: 50.6 ms   variant 6: 51.7 ms
  variant 1: 54.0 ms   variant 4: 62.5 ms   variant 7: 55.2 ms
  variant 2: 53.2 ms   variant 5: 52.5 ms   variant 8: 50.4 ms
```

So this is not a matter of picking a different `OMLX_QWEN35_Q4_MLP_VARIANT`.

## End-to-end impact

Qwen3.8-27B with a 4-bit gs128 MLP, `pp 4096 / tg 128`, MTP enabled. Same
binary and same model in both rows; the only difference is the environment
variable, and the app was relaunched between them.

| routing            | TTFT     | ppTPS | E2E    |
|--------------------|----------|-------|--------|
| default (kernel on)| 7982 ms  | 513.2 | 10.4 s |
| kernel disabled    | 4429 ms  | 924.8 | 6.8 s  |

That is a 1.80x prefill difference from the kernel alone. Decode is unchanged
(53.9 vs 53.7 tgTPS), as expected, since `seq_len <= 1` never routes.

Prompts of 1024 tokens are unaffected, consistent with the 2048-token routing
threshold: the loss begins exactly where the kernel starts being used.

With the kernel disabled, prefill scales normally with context on this model
(930 ppTPS at 4k, 819 at 8k, 765 at 16k, 615 at 32k, 547 at 64k).

## Workaround

Either of these restores full prefill speed:

```
OMLX_QWEN35_Q4_MLP=0
OMLX_QWEN35_Q4_MLP_MIN_TOKENS=999999999
```

## Suggested fix

Reject `group_size == 128` in `_is_supported_affine_linear_shape` until the
kernel handles that layout competitively. Given the gs64 numbers are at parity,
it may also be worth checking whether the patch is worth applying by default at
all on M5.

A log line when the patch declines to route, or a one-time note of which path a
model ended up on, would have made this much easier to find. The current
failure mode is a silent 2x prefill loss on the most common quantization
layout.
