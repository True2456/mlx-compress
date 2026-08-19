"""Find the ENERGY-optimal ANE fraction (not the speed-optimal one).

Speed tuning pushes toward less ANE. On battery the question is joules per
call, so the optimum can sit at a higher fraction: the ANE only has to draw
less than (its speed ratio) x GPU power to win.

Needs sudo for powermetrics:
    sudo /Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3.11 \
        artifacts/ane_probes/ane_energy_sweep.py

DRAM traffic is identical across configurations, so this measures the compute
term only -- which is the only term the ANE can change.
"""
import os, re, subprocess, sys, time
import mlx.core as mx, mlx.nn as nn
from omlx.patches import qwen35_ane_prefill as A
from omlx.patches.qwen35_q4_mlp import (
    apply_qwen35_q4_mlp_patch, apply_qwen35_q4_prefill_linear_patch)

apply_qwen35_q4_mlp_patch(); apply_qwen35_q4_prefill_linear_patch()
S, H, I, GS = 4096, 5120, 17408, 64

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(H, I, bias=False, group_size=GS, bits=4)
        self.up_proj   = nn.QuantizedLinear(H, I, bias=False, group_size=GS, bits=4)
        self.down_proj = nn.QuantizedLinear(I, H, bias=False, group_size=GS, bits=4)

mlp = MLP(); mlp.set_dtype(mx.float16); mx.eval(mlp.parameters())
x = mx.random.normal((1, S, H)).astype(mx.float16); mx.eval(x)

def sample_power(seconds):
    """Return (ane_mW, gpu_mW, cpu_mW) averaged over the window."""
    p = subprocess.run(
        ["powermetrics", "--samplers", "ane_power,gpu_power,cpu_power",
         "-i", "500", "-n", str(max(1, int(seconds * 2)))],
        capture_output=True, text=True, timeout=seconds + 30)
    out = p.stdout
    def avg(pat):
        v = [float(m) for m in re.findall(pat, out)]
        return sum(v) / len(v) if v else 0.0
    return (avg(r"ANE Power:\s+(\d+)\s*mW"),
            avg(r"GPU Power:\s+(\d+)\s*mW"),
            avg(r"CPU Power:\s+(\d+)\s*mW"))

def run_for(fn, seconds=8.0):
    fn(); mx.eval(x)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        y = fn(); mx.eval(y); n += 1
    return n, time.perf_counter() - t0

def bench(label, fn):
    import threading
    box = {}
    th = threading.Thread(target=lambda: box.update(p=sample_power(8.0)))
    th.start(); n, dt = run_for(fn, 8.0); th.join()
    ane, gpu, cpu = box.get("p", (0, 0, 0))
    ms = dt / n * 1000
    mj = (ane + gpu + cpu) * (dt / n)          # mW * s = mJ per call
    print(f"{label:26s} {ms:7.2f} ms  ANE {ane:6.0f} mW  GPU {gpu:6.0f} mW  "
          f"-> {mj:7.1f} mJ/call", flush=True)
    return mj

if os.geteuid() != 0:
    sys.exit("must run under sudo for powermetrics")

print(f"{'config':26s} {'time':>10}  {'ANE':>11}  {'GPU':>11}   energy", flush=True)
base = bench("GPU-only", lambda: mlp.down_proj(nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x)))
for frac in (0.30, 0.40, 0.53, 0.65, 0.80):
    cfg = A._AnePrefillConfig(sequence_length=S, fraction=frac, variant=8, dual_ane=True)
    mlp._omlx_ane_prefill_cache = {}
    for a in ("_omlx_ane_prefill_state", "_omlx_ane_prefill_failed"):
        if hasattr(mlp, a): delattr(mlp, a)
    mlp._omlx_ane_prefill_config = cfg
    st = A._compile_pair(mlp, cfg)
    if st is None or A._backend(mlp, x) is None:
        print(f"  fraction={frac}: ineligible"); continue
    mlp._omlx_ane_prefill_state = st
    mj = bench(f"ANE fraction={frac:.2f}", lambda: A._backend(mlp, x))
    print(f"{'':26s} {'':10}  {'':11}  {'':11}   {base/mj:5.2f}x energy vs GPU-only", flush=True)
