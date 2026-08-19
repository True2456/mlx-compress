"""Is the dynamic-weight path what caps the ANE at ~7 TFLOP/s?

Chaining surface-to-surface recovered only 3%, so the staging memcpy is not
the bottleneck. The remaining suspect is the dynamic MIL: a `wimg` feature-map
input plus an in-graph reshape, versus weights baked as compiled constants.

Same shape, same S, three programs: baked-int8, baked-fp16, dynamic.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/Users/true/AppleLLM/q38_native_engine")
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, AneDynamicLinear, _iosurface_view

S, IN, OUT = 1024, 4096, 2048
FLOPS = 2 * IN * OUT * S
W32 = (np.random.default_rng(0).standard_normal((OUT, IN)) * 0.02).astype(np.float32)
W16 = W32.astype(np.float16)
xp  = np.ascontiguousarray((np.random.default_rng(1).standard_normal((IN, S)) * 0.05).astype(np.float16))

def med(fn, n=11):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort(); return ts[len(ts) // 2]

def report(label, ms):
    print(f"  {label:22s} {ms:7.2f} ms   {FLOPS/(ms/1000)/1e12:6.2f} TFLOP/s", flush=True)
    return ms

print(f"{OUT}x{IN} projection, S={S}, {FLOPS/1e9:.1f} GFLOP\n", flush=True)
eng = AneEngine()
# AneDynamicLinear.compile initialises the module-level IOSurface handle; the
# baked path assumes it already exists, so build the dynamic program first.
dyn = AneDynamicLinear.compile(IN, OUT, S)

for label, kw in (("baked int8", dict(quantized=True)), ("baked fp16", dict(quantized=False))):
    try:
        prog = eng.compile_linear(W32, S, **kw)
        if prog is None:
            print(f"  {label:22s} compile returned None", flush=True); continue
        eng._ensure_io(prog)          # surfaces are allocated lazily
        report(label, med(lambda p=prog: eng.evaluate(p, xp, planar=True, as_float32=False)))
    except Exception as ex:
        print(f"  {label:22s} FAILED: {type(ex).__name__}: {str(ex)[:80]}", flush=True)

if dyn is not None:
    dyn.write_weight(W16)
    with _iosurface_view(dyn._x_surf, (IN, S), np.float16) as dst:
        np.copyto(dst, xp)
    report("dynamic (submit only)", med(dyn.submit))
    report("dynamic (planar API)", med(lambda: dyn.evaluate(xp, planar=True, as_float32=False)))
