"""ANE MIL op-set prober.

Builds an _ANEInMemoryModelDescriptor from candidate MIL text and reports
whether the ANE runtime accepts the program. Template and argument schema were
captured from a live oMLX compile (see docs/ANE-M5-MAX.md sec.11).
"""
import ctypes, sys, json

objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
ctypes.cdll.LoadLibrary("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")

objc.objc_getClass.restype = ctypes.c_void_p; objc.objc_getClass.argtypes=[ctypes.c_char_p]
objc.sel_registerName.restype = ctypes.c_void_p; objc.sel_registerName.argtypes=[ctypes.c_char_p]
def SEL(n): return objc.sel_registerName(n.encode())

def msg(obj, sel, *args, restype=ctypes.c_void_p, argtypes=()):
    f = objc.objc_msgSend
    f.restype = restype
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + list(argtypes)
    return f(obj, SEL(sel), *args)

NSString = objc.objc_getClass(b"NSString")
NSData   = objc.objc_getClass(b"NSData")
NSNumber = objc.objc_getClass(b"NSNumber")
NSMutableDictionary = objc.objc_getClass(b"NSMutableDictionary")
DESC = objc.objc_getClass(b"_ANEInMemoryModelDescriptor")
MODEL = objc.objc_getClass(b"_ANEInMemoryModel")

def nsstr(s):
    return msg(NSString, "stringWithUTF8String:", s.encode(),
               argtypes=[ctypes.c_char_p])
def nsdata(b):
    return msg(NSData, "dataWithBytes:length:", b, len(b),
               argtypes=[ctypes.c_char_p, ctypes.c_ulonglong])
def nsnum(i):
    return msg(NSNumber, "numberWithLongLong:", i, argtypes=[ctypes.c_longlong])
def nsdict(pairs):
    d = msg(msg(NSMutableDictionary, "alloc"), "init")
    for k, v in pairs:
        msg(d, "setObject:forKey:", v, k,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p])
    return d

BUILDINFO = ('[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
             '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
             '{"coremltools-version", "9.0"}})]')

def program(body, out="y", sig="tensor<fp16, [1, 64, 1, 1024]> x"):
    return f"program(1.3)\n{BUILDINFO}\n{{\n  func main<ios18>({sig}) {{\n{body}\n  }} -> ({out});\n}}\n"

def try_mil(mil_text, weights=()):
    w = nsdict([(nsstr(name),
                 nsdict([(nsstr("data"), nsdata(blob)), (nsstr("offset"), nsnum(0))]))
                for name, blob in weights])
    d = msg(DESC, "modelWithMILText:weights:optionsPlist:",
            nsdata(mil_text.encode()), w, nsdata(b""),
            argtypes=[ctypes.c_void_p]*3)
    if not d:
        return (False, "descriptor nil")
    m = msg(MODEL, "inMemoryModelWithDescriptor:", d, argtypes=[ctypes.c_void_p])
    if not m:
        return (False, "model nil")
    err = ctypes.c_void_p(0)
    copts = nsdict([(nsstr("kANEFAneInstanceHint"), nsnum(1)),
                    (nsstr("kANEFProcedureVariantHint"), nsnum(1))])
    ok = msg(m, "compileWithQoS:options:error:", 21, copts, ctypes.byref(err),
             restype=ctypes.c_bool,
             argtypes=[ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p])
    if ok:
        return (True, "")
    reason = "compile failed"
    if err and err.value:
        ds = msg(err.value, "localizedDescription")
        cp = msg(ds, "UTF8String", restype=ctypes.c_char_p)
        if cp: reason = cp.decode(errors="replace")[:160]
    return (False, reason)

# ---- validate the harness on a weightless program ----
T0 = "tensor<fp16, [1, 64, 1, 1024]>"
selftest = program(f'    {T0} y = relu(x = x)[name = string("t")];')
ok, why = try_mil(selftest)
print(f"harness self-test (weightless relu): {'OK' if ok else 'FAIL: '+why}")
if not ok:
    print("harness invalid; aborting sweep"); sys.exit(1)

# ---- sweep unary / binary elementwise ops on the activation path ----
T = "tensor<fp16, [1, 64, 1, 1024]>"
UNARY = ["relu","sigmoid","tanh","silu","gelu","softplus","erf","exp","log","sqrt","rsqrt",
         "abs","sign","floor","ceil","round","square","identity","cos","sin","clip",
         "leaky_relu","elu","softmax","reduce_mean","reduce_max","reduce_sum","l2_norm",
         "layer_norm","batch_norm","transpose","reshape","cast","relu6","logical_not"]
BINARY = ["add","sub","mul","real_div","maximum","minimum","pow","greater","less","equal",
          "floor_div","mod","logical_and","matmul"]

res = {"unary": {}, "binary": {}}
for op in UNARY:
    body = f'    {T} y = {op}(x = x)[name = string("t")];'
    try:
        o, why = try_mil(program(body)); res["unary"][op] = True if o else why
    except Exception as e: res["unary"][op] = f"ERR {e}"
for op in BINARY:
    body = f'    {T} y = {op}(x = x, y = x)[name = string("t")];'
    try:
        o, why = try_mil(program(body)); res["binary"][op] = True if o else why
    except Exception as e: res["binary"][op] = f"ERR {e}"

for kind in ("unary", "binary"):
    acc = sorted(k for k, v in res[kind].items() if v is True)
    rej = sorted(k for k, v in res[kind].items() if v is not True)
    print(f"\n{kind} accepted ({len(acc)}): {', '.join(acc)}")
    print(f"{kind} rejected ({len(rej)}): {', '.join(rej)}")
open("ane_opsweep_result.json","w").write(json.dumps(res, indent=2))
