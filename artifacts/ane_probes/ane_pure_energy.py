"""Pure ANE vs pure GPU on the SAME matmul: time, power, energy.

The split benchmarks were misleading for battery: they keep the GPU powered
and busy on its half, so it never idles. This runs one engine at a time.

    O=/Applications/oMLX.app/Contents/Resources
    sudo env PYTHONPATH="$O/Python/framework-mlx-base/lib/python3.11/site-packages:$O:$HOME/AppleLLM/q38_native_engine" \
        "$O/Python/cpython-3.11/bin/python3.11" artifacts/ane_probes/ane_pure_energy.py
"""
import os, re, subprocess, sys, threading, time
import numpy as np, mlx.core as mx
from runtime.q38_ane_engine import AneDynamicLinear

if os.geteuid() != 0:
    sys.exit("needs sudo for powermetrics")

S, IN, OUT = 1024, 4096, 2048          # one Qwen-ish projection
SECS = 8.0

def power(seconds):
    p = subprocess.run(["powermetrics", "--samplers", "ane_power,gpu_power,cpu_power",
                        "-i", "500", "-n", str(max(1, int(seconds * 2)))],
                       capture_output=True, text=True, timeout=seconds + 30)
    def avg(pat):
        v = [float(m) for m in re.findall(pat, p.stdout)]
        return sum(v) / len(v) if v else 0.0
    return avg(r"ANE Power:\s+(\d+)\s*mW"), avg(r"GPU Power:\s+(\d+)\s*mW"), avg(r"CPU Power:\s+(\d+)\s*mW")

def bench(label, fn):
    fn()
    box = {}
    th = threading.Thread(target=lambda: box.update(p=power(SECS))); th.start()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < SECS:
        fn(); n += 1
    dt = time.perf_counter() - t0; th.join()
    ane, gpu, cpu = box.get("p", (0, 0, 0))
    ms = dt / n * 1000
    mj = (ane + gpu + cpu) * (dt / n)
    flops = 2 * IN * OUT * S
    print(f"{label:14s} {ms:8.2f} ms  ANE {ane:6.0f} mW  GPU {gpu:6.0f} mW  "
          f"CPU {cpu:5.0f} mW  {mj:8.2f} mJ/call  {flops/(ms/1000)/1e12:5.2f} TFLOP/s", flush=True)
    return ms, mj

print(f"one {OUT}x{IN} projection over {S} tokens\n", flush=True)

# --- GPU ---
W = mx.random.normal((OUT, IN)).astype(mx.float16)
x = mx.random.normal((S, IN)).astype(mx.float16); mx.eval(W, x)
def gpu():
    y = x @ W.T; mx.eval(y)
g_ms, g_mj = bench("GPU", gpu)

# --- ANE (weight resident, no per-call copy) ---
prog = AneDynamicLinear.compile(IN, OUT, S)
if prog is None:
    sys.exit("ANE compile failed")
Wn = np.ascontiguousarray(np.array(W).astype(np.float16))
xn = np.ascontiguousarray(np.array(x).astype(np.float16))
prog.write_weight(Wn)
def ane():
    prog.evaluate(xn)
a_ms, a_mj = bench("ANE (pure)", ane)

# planar fp16 in and out: no transpose, no fp32 widening
xp = np.ascontiguousarray(xn.T)
def ane_planar():
    prog.evaluate(xp, planar=True, as_float32=False)
p_ms, p_mj = bench("ANE (planar)", ane_planar)

print(f"\nANE      vs GPU:  {g_ms/a_ms:5.2f}x speed   {g_mj/a_mj:5.2f}x energy")
print(f"ANE planar vs GPU:  {g_ms/p_ms:5.2f}x speed   {g_mj/p_mj:5.2f}x energy")
print("energy >1.0 means the ANE uses less per call")
