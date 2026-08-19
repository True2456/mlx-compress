"""Does chaining ANE layers surface-to-surface recover the missing headroom?

Pure-ANE measured 7.60 TFLOP/s against 13.2 in the hybrid region. The gap is
~1 ms/call of memcpy staging activations through numpy. In a chain, layer N's
output is layer N+1's input — both already in IOSurfaces — so the staging
should only be paid at the ends.

Compares three ways of moving activations between layers:
  numpy   : evaluate() -> numpy -> evaluate()      (what a per-layer swap costs)
  surface : memmove y_surf -> next x_surf          (no numpy, still one copy)
"""
import ctypes, sys, time
import numpy as np
sys.path.insert(0, "/Users/true/AppleLLM/q38_native_engine")
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneDynamicLinear, _iosurface_view

S, D, N = 1024, 2048, 4          # square layers so output feeds input directly

progs = []
for i in range(N):
    p = AneDynamicLinear.compile(D, D, S)
    if p is None:
        sys.exit("compile failed")
    p.write_weight((np.random.default_rng(i).standard_normal((D, D)) * 0.02).astype(np.float16))
    progs.append(p)
print(f"{N} chained {D}x{D} layers, S={S}", flush=True)

x0 = np.ascontiguousarray(
    (np.random.default_rng(99).standard_normal((D, S)) * 0.05).astype(np.float16))

def med(fn, n=9):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort(); return ts[len(ts) // 2]

def via_numpy():
    h = x0
    for p in progs:
        h = p.evaluate(h, planar=True, as_float32=False)
    return h

def surf_copy(src_surf, dst_surf, nbytes):
    s1, s2 = ctypes.c_uint32(0), ctypes.c_uint32(0)
    E._iosurf.IOSurfaceLock(src_surf, 1, ctypes.byref(s1))
    E._iosurf.IOSurfaceLock(dst_surf, 0, ctypes.byref(s2))
    try:
        ctypes.memmove(E._iosurf.IOSurfaceGetBaseAddress(dst_surf),
                       E._iosurf.IOSurfaceGetBaseAddress(src_surf), nbytes)
    finally:
        E._iosurf.IOSurfaceUnlock(dst_surf, 0, ctypes.byref(s2))
        E._iosurf.IOSurfaceUnlock(src_surf, 1, ctypes.byref(s1))

def via_surface():
    """Stage in once, chain surface->surface, read out once."""
    nbytes = D * S * 2
    with _iosurface_view(progs[0]._x_surf, (D, S), np.float16) as dst:
        np.copyto(dst, x0)
    for i, p in enumerate(progs):
        p.submit()
        if i + 1 < N:
            surf_copy(p._y_surf, progs[i + 1]._x_surf, nbytes)
    with _iosurface_view(progs[-1]._y_surf, (D, S), np.float16) as src:
        return src.copy()


t_np = med(via_numpy)
per = t_np / N
print(f"  numpy chaining : {t_np:7.2f} ms total   {per:6.2f} ms/layer", flush=True)
flops = 2 * D * D * S
print(f"  -> {flops / (per / 1000) / 1e12:5.2f} TFLOP/s per layer", flush=True)
t_sf = med(via_surface); per_sf = t_sf / N
print(f"  surface chain  : {t_sf:7.2f} ms total   {per_sf:6.2f} ms/layer", flush=True)
print(f"  -> {flops / (per_sf / 1000) / 1e12:5.2f} TFLOP/s per layer   ({t_np/t_sf:.2f}x vs numpy)", flush=True)
print(f"     (single-layer planar measured 7.60; hybrid-region ceiling 13.2)")
