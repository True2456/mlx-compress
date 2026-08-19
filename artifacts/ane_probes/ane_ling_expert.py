"""Ling-shaped MoE experts on the ANE: full SwiGLU, routed by procedureIndex.

Real Ling-3.0-flash geometry: hidden 2560, moe_intermediate 768, top-8 of 512.
Each expert is gate+up+down with SwiGLU between, all inside one procedure, so
routing is an index rather than a weight swap.

Blob budget: 3 blobs per expert (gate/up/down fp16) against the measured 16
per program => 5 experts per program. 8 active experts = 2 programs/layer.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/Users/true/AppleLLM/q38_native_engine")
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, AneDynamicLinear, _iosurface_view

H, M = 2560, 768                 # hidden, moe_intermediate
S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
NEXP = int(sys.argv[2]) if len(sys.argv) > 2 else 5     # 16 blobs / 3 per expert

rng = np.random.default_rng(0)
experts = [(
    (rng.standard_normal((M, H)) * 0.02).astype(np.float16),   # gate
    (rng.standard_normal((M, H)) * 0.02).astype(np.float16),   # up
    (rng.standard_normal((H, M)) * 0.02).astype(np.float16),   # down
) for _ in range(NEXP)]

blobs, procs = {}, []
for i, (g, u, d) in enumerate(experts):
    blobs[f"g{i}.bin"] = g.reshape(M, H, 1, 1).tobytes()
    blobs[f"u{i}.bin"] = u.reshape(M, H, 1, 1).tobytes()
    blobs[f"d{i}.bin"] = d.reshape(H, M, 1, 1).tobytes()
    procs.append(f'''  func procedure{i:03d}<ios18>(tensor<fp16, [1, {H}, 1, {S}]> x) {{
    tensor<fp16, [{M}, {H}, 1, 1]> gw{i} = const()[name=string("gw{i}"), val=tensor<fp16, [{M}, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/g{i}.bin"), offset=uint64(64)))];
    tensor<fp16, [{M}, {H}, 1, 1]> uw{i} = const()[name=string("uw{i}"), val=tensor<fp16, [{M}, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/u{i}.bin"), offset=uint64(64)))];
    tensor<fp16, [{H}, {M}, 1, 1]> dw{i} = const()[name=string("dw{i}"), val=tensor<fp16, [{H}, {M}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/d{i}.bin"), offset=uint64(64)))];
    string pt{i} = const()[name=string("pt{i}"), val=string("valid")];
    tensor<int32, [2]> st{i} = const()[name=string("st{i}"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd{i} = const()[name=string("pd{i}"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl{i} = const()[name=string("dl{i}"), val=tensor<int32, [2]>([1,1])];
    int32 gr{i} = const()[name=string("gr{i}"), val=int32(1)];
    tensor<fp16, [1, {M}, 1, {S}]> g{i} = conv(dilations=dl{i}, groups=gr{i}, pad=pd{i}, pad_type=pt{i}, strides=st{i}, weight=gw{i}, x=x)[name=string("gate{i}")];
    tensor<fp16, [1, {M}, 1, {S}]> sg{i} = sigmoid(x=g{i})[name=string("sig{i}")];
    tensor<fp16, [1, {M}, 1, {S}]> si{i} = mul(x=g{i}, y=sg{i})[name=string("silu{i}")];
    tensor<fp16, [1, {M}, 1, {S}]> u{i} = conv(dilations=dl{i}, groups=gr{i}, pad=pd{i}, pad_type=pt{i}, strides=st{i}, weight=uw{i}, x=x)[name=string("up{i}")];
    tensor<fp16, [1, {M}, 1, {S}]> a{i} = mul(x=si{i}, y=u{i})[name=string("act{i}")];
    tensor<fp16, [1, {H}, 1, {S}]> y{i} = conv(dilations=dl{i}, groups=gr{i}, pad=pd{i}, pad_type=pt{i}, strides=st{i}, weight=dw{i}, x=a{i})[name=string("down{i}")];
  }} -> (y{i});''')

mil = f"program(1.3)\n{E._BUILD_INFO}\n{{\n" + "\n".join(procs) + "\n}\n"
mb = sum(len(b) for b in blobs.values()) / 1e6
print(f"Ling expert: h={H} m={M}, {NEXP} experts/program ({len(blobs)} blobs), "
      f"{mb:.0f} MB, S={S}", flush=True)

eng = AneEngine(); _ = AneDynamicLinear.compile(H, H, S)
t0 = time.perf_counter()
prog = eng.compile_multiproc(mil, blobs, H, H, S)
if prog is None:
    sys.exit("  COMPILE FAILED")
print(f"  compiled {NEXP} full SwiGLU experts in {time.perf_counter()-t0:.2f}s", flush=True)

eng._ensure_io(prog)
x = np.ascontiguousarray((rng.standard_normal((H, S)) * 0.05).astype(np.float16))
with _iosurface_view(prog._in_surf, (H, S), np.float16) as dst:
    np.copyto(dst, x)

def ref(i):
    g, u, d = (w.astype(np.float32) for w in experts[i])
    xf = x.astype(np.float32)
    gg = g @ xf
    return d @ ((gg / (1.0 + np.exp(-gg))) * (u @ xf))

print("  correctness:", flush=True)
for i in (0, NEXP - 1):
    eng.submit(prog, procedure_index=i)
    with _iosurface_view(prog._out_surf, (H, S), np.float16) as out:
        got = out.copy().astype(np.float32)
    r = ref(i)
    print(f"    expert {i}: rel err {np.sqrt(((got-r)**2).sum()/(r**2).sum()):.3e}", flush=True)

def med(fn, n=15):
    fn(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    ts.sort(); return ts[len(ts)//2]

ctr = {"i": 0}
def rotate():
    ctr["i"] = (ctr["i"] + 1) % NEXP
    eng.submit(prog, procedure_index=ctr["i"])
ms = med(rotate)
flops = 2 * (2 * M * H + H * M) * S          # gate + up + down
print(f"\n  routed dispatch: {ms:.3f} ms/expert   {flops/(ms/1000)/1e12:.2f} TFLOP/s", flush=True)
print(f"  a full Ling layer (8 active) ~= {ms*8:.2f} ms of ANE MLP", flush=True)
print(f"  42 layers ~= {ms*8*42:.0f} ms for {S} tokens of prefill", flush=True)
