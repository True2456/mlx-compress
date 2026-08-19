import time, json, math, sys
import mlx.core as mx
import mlx.nn as nn
from omlx.custom_kernels.qwen35_prefill import fast
from omlx.patches import qwen35_ane_prefill as A

H, I, GS = 5120, 17408, 64
out = {}

# ---------------------------------------------------------------- A. instance hints 3 / 4
print("=== A. instance hint range ===")
W = mx.contiguous(mx.random.normal((2048, H)).astype(mx.float32)); mx.eval(W)
inst_ok = {}
for inst in (0, 1, 2, 3, 4, 5):
    try:
        m = fast.qwen35_ane_compile_linear(W, 2048, inst)
        inst_ok[inst] = "compiled" if m is not None else "None"
    except Exception as e:
        inst_ok[inst] = f"ERR {type(e).__name__}: {str(e)[:60]}"
    print(f"  instance {inst}: {inst_ok[inst]}")
out["instance_hints"] = inst_ok

# ---------------------------------------------------------------- B. sequence-length support
print("\n=== B. sequence lengths that compile ===")
seq_ok = {}
for S in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
    try:
        t0 = time.perf_counter()
        m = fast.qwen35_ane_compile_linear(W, S, 1)
        seq_ok[S] = {"ok": m is not None, "compile_s": round(time.perf_counter()-t0, 3)}
    except Exception as e:
        seq_ok[S] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:70]}"}
    print(f"  S={S:6d}: {seq_ok[S]}")
out["sequence_lengths"] = seq_ok

# ---------------------------------------------------------------- C. numerics: ANE vs pure GPU
print("\n=== C. numerical fidelity (ANE+GPU hybrid vs pure GPU vs fp32 ref) ===")
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(H, I, bias=False, group_size=GS, bits=4)
        self.up_proj   = nn.QuantizedLinear(H, I, bias=False, group_size=GS, bits=4)
        self.down_proj = nn.QuantizedLinear(I, H, bias=False, group_size=GS, bits=4)

mlp = MLP(); mlp.set_dtype(mx.float16); mx.eval(mlp.parameters())

def deq(l):
    return mx.dequantize(l.weight, l.scales, l.biases, group_size=GS, bits=4)

S = 4096
x = (mx.random.normal((1, S, H)) * 0.05).astype(mx.float16); mx.eval(x)

gw, uw, dw = deq(mlp.gate_proj), deq(mlp.up_proj), deq(mlp.down_proj)
xf = x.astype(mx.float32)
g = xf @ gw.astype(mx.float32).T
u = xf @ uw.astype(mx.float32).T
ref = (g * mx.sigmoid(g) * u) @ dw.astype(mx.float32).T
mx.eval(ref)

pure_gpu = mlp.down_proj(nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
mx.eval(pure_gpu)

def rel_err(a, b):
    a = a.astype(mx.float32); b = b.astype(mx.float32)
    return float(mx.sqrt(mx.sum((a-b)**2) / mx.sum(b**2)))

res = {"pure_gpu_vs_fp32ref": rel_err(pure_gpu, ref)}
for dual in (True, False):
    cfg = A._AnePrefillConfig(sequence_length=S, fraction=0.53, variant=8, dual_ane=dual)
    mlp._omlx_ane_prefill_cache = {}
    for a_ in ("_omlx_ane_prefill_state", "_omlx_ane_prefill_failed"):
        if hasattr(mlp, a_): delattr(mlp, a_)
    mlp._omlx_ane_prefill_config = cfg
    st = A._compile_pair(mlp, cfg)
    mlp._omlx_ane_prefill_state = st
    y = A._backend(mlp, x); mx.eval(y)
    k = "dual" if dual else "single"
    res[f"ane_{k}_vs_fp32ref"] = rel_err(y, ref)
    res[f"ane_{k}_vs_pure_gpu"] = rel_err(y, pure_gpu)
for k, v in res.items(): print(f"  {k:28s} {v:.6e}")
out["numerics"] = res

# ---------------------------------------------------------------- D. effective throughput
print("\n=== D. ANE effective throughput on the gate+up slice ===")
st = mlp._omlx_ane_prefill_state
ane_rows = st.ane_outputs * 2          # gate + up stacked
flops = 2 * ane_rows * H * S
fast.qwen35_ane_profile_reset(); fast.qwen35_ane_profile_set_enabled(True)
for _ in range(5):
    y = A._backend(mlp, x); mx.eval(y)
snap = fast.qwen35_ane_profile_snapshot().get("mlp", {})
fast.qwen35_ane_profile_set_enabled(False)
ops = snap.get("operations", 0) or 1
region_s = snap.get("ane_region_ns", 0) / 1e9 / ops
e0 = snap.get("ane0_eval_ns", 0)/1e9/ops; e1 = snap.get("ane1_eval_ns", 0)/1e9/ops
print(f"  ane rows={ane_rows} K={H} S={S}  FLOPs/call={flops/1e9:.1f} G")
print(f"  region {region_s*1e3:.2f} ms  -> {flops/region_s/1e12:.2f} TFLOP/s (fp16, ANE region)")
if e0: print(f"  ane0 {e0*1e3:.2f} ms  ane1 {e1*1e3:.2f} ms")
out["throughput"] = {"flops_per_call": flops, "region_s": region_s,
                     "tflops": flops/region_s/1e12 if region_s else None,
                     "profile": snap}

print("\n=== RESULT ===")
print(json.dumps(out, indent=2, default=float))
