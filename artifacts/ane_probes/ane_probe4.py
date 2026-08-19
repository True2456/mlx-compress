import time, json
import mlx.core as mx
import mlx.nn as nn
from omlx.custom_kernels.qwen35_prefill import fast

H, S, GS = 5120, 4096, 64
ANE_ROWS = 9216          # per bank-pair half, mirrors production ane_outputs
out = {}

x = (mx.random.normal((1, S, H)) * 0.05).astype(mx.float16); mx.eval(x)

# GPU remainder, quantized (mirrors _compile_pair: gate+up rows past ane_outputs)
w_gpu = mx.contiguous(mx.random.normal((2 * 8192, H)).astype(mx.float16))
gw, gs_, gb = mx.quantize(w_gpu, group_size=GS, bits=4); mx.eval(gw, gs_, gb)

half = ANE_ROWS // 2
d0 = mx.contiguous(mx.random.normal((2 * half, H)).astype(mx.float32))
d1 = mx.contiguous(mx.random.normal((2 * half, H)).astype(mx.float32))
mx.eval(d0, d1)

def measure(i0, i1, n=7):
    m0 = fast.qwen35_ane_compile_linear(d0, S, i0)
    m1 = fast.qwen35_ane_compile_linear(d1, S, i1)
    fn = lambda: fast.qwen35_ane_dual_q4_swiglu_t(x, gw, gs_, gb, m0, m1, 8, GS)
    y = fn(); mx.eval(y)                                   # warmup
    fast.qwen35_ane_profile_reset(); fast.qwen35_ane_profile_set_enabled(True)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); y = fn(); mx.eval(y)
        ts.append((time.perf_counter()-t)*1e3)
    snap = fast.qwen35_ane_profile_snapshot().get("mlp", {})
    fast.qwen35_ane_profile_set_enabled(False)
    ops = snap.get("operations", 0) or 1
    e0 = snap.get("ane0_eval_ns",0)/1e6/ops; e1 = snap.get("ane1_eval_ns",0)/1e6/ops
    reg = snap.get("ane_region_ns",0)/1e6/ops; gpu = snap.get("gpu_qmm_ns",0)/1e6/ops
    ts.sort()
    return {"pair": f"{i0},{i1}", "median_ms": round(ts[len(ts)//2],2),
            "ane0_ms": round(e0,2), "ane1_ms": round(e1,2),
            "region_ms": round(reg,2), "gpu_ms": round(gpu,2),
            "overlap_x": round((e0+e1)/reg, 3) if reg else None}

print("=== E. do instance hints 3 and 4 also run in parallel? ===")
rows = []
for pair in ((1,2), (3,4), (1,3), (2,4), (1,1), (3,3)):
    try:
        r = measure(*pair); rows.append(r)
        print(f"  hints {r['pair']:4s}  median {r['median_ms']:6.2f} ms  "
              f"ane0 {r['ane0_ms']:6.2f}  ane1 {r['ane1_ms']:6.2f}  "
              f"region {r['region_ms']:6.2f}  overlap {r['overlap_x']}")
    except Exception as e:
        print(f"  hints {pair}: ERR {e}"); rows.append({"pair": str(pair), "err": str(e)})
out["instance_pairs"] = rows

print("\n=== F. effective ANE throughput ===")
best = min((r for r in rows if "region_ms" in r), key=lambda r: r["region_ms"])
flops = 2 * (2 * ANE_ROWS) * H * S          # gate+up rows stacked
reg_s = best["region_ms"]/1e3
print(f"  ane rows={2*ANE_ROWS} K={H} S={S}")
print(f"  {flops/1e9:.1f} GFLOP per call, region {best['region_ms']:.2f} ms (hints {best['pair']})")
print(f"  -> {flops/reg_s/1e12:.2f} TFLOP/s fp16 across the ANE region")
print(f"  GPU half in the same call: {best['gpu_ms']:.2f} ms")
out["throughput"] = {"gflops_per_call": flops/1e9, "best": best,
                     "tflops": flops/reg_s/1e12}

print("\n=== RESULT ===")
print(json.dumps(out, indent=2, default=float))
