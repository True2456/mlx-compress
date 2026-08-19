"""Can one blob hold many experts, addressed by BLOBFILE offset?

The compiler accepts at most 16 blobs per program, which would cap a baked
MoE layer at 16 experts (fp16) or 8 (int8 + scale). But BLOBFILE takes an
offset, so N experts can share ONE blob at N different offsets — making the
blob limit irrelevant and the procedure count the only question.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/Users/true/AppleLLM/q38_native_engine")
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, AneDynamicLinear, _iosurface_view

S, IN, OUT = 512, 1024, 1024
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
HDR = 64                       # blob header; payload starts here

rng = np.random.default_rng(0)
W = [(rng.standard_normal((OUT, IN)) * 0.02).astype(np.float16) for _ in range(N)]
stride = OUT * IN * 2          # bytes per expert, fp16

# one payload: expert i at HDR + i*stride
payload = b"".join(w.reshape(OUT, IN, 1, 1).tobytes() for w in W)

procs = []
for i in range(N):
    off = HDR + i * stride
    procs.append(f'''  func procedure{i:03d}<ios18>(tensor<fp16, [1, {IN}, 1, {S}]> x) {{
    tensor<fp16, [{OUT}, {IN}, 1, 1]> w = const()[name=string("w{i}"), val=tensor<fp16, [{OUT}, {IN}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/experts.bin"), offset=uint64({off})))];
    string pt{i} = const()[name=string("pt{i}"), val=string("valid")];
    tensor<int32, [2]> st{i} = const()[name=string("st{i}"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd{i} = const()[name=string("pd{i}"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl{i} = const()[name=string("dl{i}"), val=tensor<int32, [2]>([1,1])];
    int32 gr{i} = const()[name=string("gr{i}"), val=int32(1)];
    tensor<fp16, [1, {OUT}, 1, {S}]> y{i} = conv(dilations=dl{i}, groups=gr{i}, pad=pd{i}, pad_type=pt{i}, strides=st{i}, weight=w, x=x)[name=string("conv{i}")];
  }} -> (y{i});''')
mil = f"program(1.3)\n{E._BUILD_INFO}\n{{\n" + "\n".join(procs) + "\n}\n"

print(f"{N} procedures sharing ONE blob ({len(payload)/1e6:.0f} MB payload), MIL {len(mil)/1024:.0f} KB", flush=True)
eng = AneEngine(); _ = AneDynamicLinear.compile(IN, OUT, S)
t0 = time.perf_counter()
prog = eng.compile_multiproc(mil, {"experts.bin": payload}, IN, OUT, S)
if prog is None:
    sys.exit("  COMPILE FAILED")
print(f"  compiled in {time.perf_counter()-t0:.1f}s", flush=True)

eng._ensure_io(prog)
x = np.ascontiguousarray((rng.standard_normal((IN, S)) * 0.05).astype(np.float16))
with _iosurface_view(prog._in_surf, (IN, S), np.float16) as dst:
    np.copyto(dst, x)
ok = True
for idx in (0, N // 2, N - 1):
    if not eng.submit(prog, procedure_index=idx):
        print(f"    idx {idx}: submit failed"); ok = False; continue
    with _iosurface_view(prog._out_surf, (OUT, S), np.float16) as out:
        got = out.copy().astype(np.float32)
    ref = W[idx].astype(np.float32) @ x.astype(np.float32)
    rel = np.sqrt(((got - ref) ** 2).sum() / (ref ** 2).sum())
    flag = "OK" if rel < 0.05 else "MISMATCH"
    if rel >= 0.05: ok = False
    print(f"    idx {idx:3d}: rel err {rel:.3e} {flag}", flush=True)
print("  => offset packing works" if ok else "  => offset packing NOT correct")
