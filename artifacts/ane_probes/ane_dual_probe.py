import time, json, sys
import mlx.core as mx
import mlx.nn as nn
from omlx.custom_kernels.qwen35_prefill import fast
from omlx.patches import qwen35_ane_prefill as A

S, H, I, GS = 4096, 5120, 17408, 64

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(H, I, bias=False, group_size=GS, bits=4)
        self.up_proj   = nn.QuantizedLinear(H, I, bias=False, group_size=GS, bits=4)
        self.down_proj = nn.QuantizedLinear(I, H, bias=False, group_size=GS, bits=4)

mlp = MLP()
mlp.set_dtype(mx.float16)
mx.eval(mlp.parameters())
print("eligible_pair:", A._eligible_pair(mlp))

x = mx.random.normal((1, S, H)).astype(mx.float16); mx.eval(x)

def bench(dual, n=7):
    cfg = A._AnePrefillConfig(sequence_length=S, fraction=0.53, variant=8, dual_ane=dual)
    mlp._omlx_ane_prefill_cache = {}
    for a in ("_omlx_ane_prefill_state", "_omlx_ane_prefill_failed"):
        if hasattr(mlp, a): delattr(mlp, a)
    mlp._omlx_ane_prefill_config = cfg

    t0 = time.perf_counter()
    st = A._compile_pair(mlp, cfg)
    compile_s = time.perf_counter() - t0
    if st is None:
        return {"error": "_compile_pair returned None"}
    mlp._omlx_ane_prefill_state = st
    print(f"  ane_outputs={st.ane_outputs} gpu_outputs={st.gpu_outputs} "
          f"model1={'yes' if st.model1 is not None else 'no'} compile={compile_s:.1f}s")

    y = A._backend(mlp, x)
    if y is None:
        return {"error": "_backend returned None (input ineligible)"}
    mx.eval(y)

    fast.qwen35_ane_profile_reset(); fast.qwen35_ane_profile_set_enabled(True)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); y = A._backend(mlp, x); mx.eval(y)
        ts.append((time.perf_counter() - t) * 1e3)
    snap = fast.qwen35_ane_profile_snapshot()
    fast.qwen35_ane_profile_set_enabled(False)
    ts.sort()
    return {"median_ms": ts[len(ts)//2], "all_ms": [round(v,2) for v in ts],
            "compile_s": compile_s, "ane_outputs": st.ane_outputs,
            "profile_mlp": snap.get("mlp", {})}

out = {}
for name, dual in (("dual", True), ("single", False)):
    print(f"\n--- {name} (dual_ane={dual}) ---")
    try:
        out[name] = bench(dual)
        r = out[name]
        if "median_ms" in r:
            print(f"  median {r['median_ms']:.2f} ms   runs={r['all_ms']}")
            p = r["profile_mlp"]
            for k in ("ane_region_ns","ane0_eval_ns","ane1_eval_ns",
                      "ane0_launch_ns","ane1_launch_ns","gpu_qmm_ns","operations"):
                if k in p: print(f"    {k:16s} {p[k]}")
        else:
            print(" ", r)
    except Exception as e:
        out[name] = {"error": repr(e)}; print("  FAILED:", repr(e))

print("\n=== RESULT ===")
print(json.dumps(out, indent=2, default=float))
