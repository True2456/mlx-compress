"""Gating probe for baked-expert MoE on the ANE.

Three questions, in order:
  1. Does a multi-procedure program compile at all?
  2. Does procedureIndex actually select the right expert?
  3. Does switching index re-DMA weights, or are they all resident?

If (3) shows a cost proportional to weight size, the design collapses back to
the dynamic path. If switching is flat, one program per layer holds every
expert and routing is just an index.
"""
import ctypes, sys, time
import numpy as np
sys.path.insert(0, "/Users/true/AppleLLM/q38_native_engine")
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, AneDynamicLinear, _iosurface_view

S, IN, OUT = 512, 1024, 1024      # Ling-ish expert scale, small enough to iterate
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
FP16 = len(sys.argv) > 2 and sys.argv[2] == "fp16"   # 1 blob/proc instead of 2

def quant(w):
    sc = np.abs(w).max(axis=1, keepdims=True).astype(np.float32)
    sc = np.where(sc == 0, 1.0, sc)
    q = np.clip(np.round(w / sc * 127.0), -128, 127).astype(np.int8)
    s16 = (sc.squeeze() / 127.0).astype(np.float16)
    return q, s16, q.astype(np.float32) * s16.astype(np.float32)[:, None]

def gen_mil(n):
    procs = []
    for i in range(n):
        if FP16:
            wdecl = (f'    tensor<fp16, [{OUT}, {IN}, 1, 1]> w = const()[name=string("w{i}"), '
                     f'val=tensor<fp16, [{OUT}, {IN}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w{i}.bin"), offset=uint64(64)))];')
        else:
            wdecl = (f'    tensor<int8, [{OUT}, {IN}, 1, 1]> wd = const()[name=string("wd{i}"), '
                     f'val=tensor<int8, [{OUT}, {IN}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w{i}.bin"), offset=uint64(64)))];\n'
                     f'    tensor<fp16, [{OUT}, 1, 1, 1]> ws = const()[name=string("ws{i}"), '
                     f'val=tensor<fp16, [{OUT}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/s{i}.bin"), offset=uint64(64)))];\n'
                     f'    tensor<fp16, [{OUT}, {IN}, 1, 1]> w = constexpr_blockwise_shift_scale(data=wd, scale=ws)[name=string("dq{i}")];')
        procs.append(f"""  func procedure{i:03d}<ios18>(tensor<fp16, [1, {IN}, 1, {S}]> x) {{
{wdecl}
    string pt{i} = const()[name=string("pt{i}"), val=string("valid")];
    tensor<int32, [2]> st{i} = const()[name=string("st{i}"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd{i} = const()[name=string("pd{i}"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl{i} = const()[name=string("dl{i}"), val=tensor<int32, [2]>([1,1])];
    int32 gr{i} = const()[name=string("gr{i}"), val=int32(1)];
    tensor<fp16, [1, {OUT}, 1, {S}]> y{i} = conv(dilations=dl{i}, groups=gr{i}, pad=pd{i}, pad_type=pt{i}, strides=st{i}, weight=w, x=x)[name=string("conv{i}")];
  }} -> (y{i});""")
    return f"program(1.3)\n{E._BUILD_INFO}\n{{\n" + "\n".join(procs) + "\n}\n"

eng = AneEngine()
_ = AneDynamicLinear.compile(IN, OUT, S)     # initialises the IOSurface handle

rng = np.random.default_rng(0)
W = [(rng.standard_normal((OUT, IN)) * 0.02).astype(np.float32) for _ in range(NPROC)]
blobs, refs = {}, []
for i, w in enumerate(W):
    q, s16, deq = quant(w); refs.append(deq)
    if FP16:
        blobs[f"w{i}.bin"] = w.astype(np.float16).reshape(OUT, IN, 1, 1).tobytes()
        refs[-1] = w.astype(np.float16).astype(np.float32)
    else:
        blobs[f"w{i}.bin"] = q.reshape(OUT, IN, 1, 1).tobytes()
        blobs[f"s{i}.bin"] = s16.reshape(OUT, 1, 1, 1).tobytes()

mil = gen_mil(NPROC)
mb = sum(len(x) for x in (b"",)) + NPROC * (OUT*IN + OUT*2) / 1e6
print(f"{NPROC} procedures, {OUT}x{IN} each, ~{mb:.0f} MB of weights, MIL {len(mil)/1024:.0f} KB", flush=True)

t0 = time.perf_counter()
prog = eng.compile_multiproc(mil, blobs, IN, OUT, S)
if prog is None:
    sys.exit("  COMPILE FAILED")
print(f"  compiled in {time.perf_counter()-t0:.1f}s", flush=True)

eng._ensure_io(prog)
x = np.ascontiguousarray((rng.standard_normal((IN, S)) * 0.05).astype(np.float16))
with _iosurface_view(prog._in_surf, (IN, S), np.float16) as dst:
    np.copyto(dst, x)

def run(idx):
    E._rebuild_request(prog, idx) if hasattr(E, "_rebuild_request") else None
    return eng.submit(prog, procedure_index=idx) if "procedure_index" in eng.submit.__code__.co_varnames else None

print("\n  checking procedureIndex selects the right expert:", flush=True)
for idx in (0, NPROC // 2, NPROC - 1):
    ok = run(idx)
    if ok is None:
        print(f"    idx {idx}: engine.submit() has no procedure_index parameter yet")
        break
    with _iosurface_view(prog._out_surf, (OUT, S), np.float16) as out:
        got = out.copy().astype(np.float32)
    ref = refs[idx].astype(np.float32) @ x.astype(np.float32)
    rel = np.sqrt(((got - ref) ** 2).sum() / (ref ** 2).sum())
    print(f"    idx {idx:3d}: rel err {rel:.3e} {'OK' if rel < 0.05 else 'MISMATCH'}", flush=True)

# --- Q3: does switching procedure re-DMA weights, or are they all resident? ---
print("\n  dispatch cost, same index vs rotating:", flush=True)
def med(fn, n=25):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    ts.sort(); return ts[len(ts)//2]

same = med(lambda: eng.submit(prog, procedure_index=0))
ctr = {"i": 0}
def rotate():
    ctr["i"] = (ctr["i"] + 1) % NPROC
    eng.submit(prog, procedure_index=ctr["i"])
rot = med(rotate)
flops = 2 * IN * OUT * S
print(f"    fixed index   {same:6.3f} ms   {flops/(same/1000)/1e12:5.2f} TFLOP/s", flush=True)
print(f"    rotating      {rot:6.3f} ms   {flops/(rot/1000)/1e12:5.2f} TFLOP/s   ({rot/same:.2f}x)", flush=True)
print("    (>>1x means switching re-DMAs weights and the design collapses)", flush=True)
