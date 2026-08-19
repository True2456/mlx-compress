import ctypes, os
objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
ctypes.cdll.LoadLibrary("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")
for n,r,a in [("objc_getClass",ctypes.c_void_p,[ctypes.c_char_p]),
              ("object_getClass",ctypes.c_void_p,[ctypes.c_void_p]),
              ("sel_registerName",ctypes.c_void_p,[ctypes.c_char_p]),
              ("class_getInstanceMethod",ctypes.c_void_p,[ctypes.c_void_p,ctypes.c_void_p]),
              ("class_getName",ctypes.c_char_p,[ctypes.c_void_p]),
              ("method_getImplementation",ctypes.c_void_p,[ctypes.c_void_p]),
              ("method_setImplementation",ctypes.c_void_p,[ctypes.c_void_p,ctypes.c_void_p])]:
    f=getattr(objc,n); f.restype=r; f.argtypes=a

def msg(o,s,*args,restype=ctypes.c_void_p,argtypes=()):
    f=objc.objc_msgSend; f.restype=restype
    f.argtypes=[ctypes.c_void_p,ctypes.c_void_p]+list(argtypes)
    return f(o,objc.sel_registerName(s.encode()),*args)
def desc(o):
    if not o: return "(nil)"
    cp = msg(msg(o,"description"),"UTF8String",restype=ctypes.c_char_p)
    return cp.decode(errors="replace") if cp else "(?)"

sel = objc.sel_registerName(b"compileWithQoS:options:error:")
hooked=[]
for cname in (b"_ANEInMemoryModel", b"_ANEModel"):
    cls = objc.objc_getClass(cname)
    if not cls: continue
    m = objc.class_getInstanceMethod(cls, sel)
    if not m: continue
    orig = objc.method_getImplementation(m)
    IMP = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)
    def mk(orig=orig, cname=cname, IMP=IMP):
        @IMP
        def hook(self_, cmd, qos, opts, err):
            print(f"=== {cname.decode()} compileWithQoS: qos={qos} ===", flush=True)
            print("options:", desc(opts), flush=True)
            r = ctypes.cast(orig, IMP)(self_, cmd, qos, opts, err)
            print("returned:", r, flush=True)
            return r
        return hook
    h = mk(); hooked.append(h)
    objc.method_setImplementation(m, ctypes.cast(h, ctypes.c_void_p))
    print("hooked", cname.decode(), flush=True)

import mlx.core as mx
from omlx.custom_kernels.qwen35_prefill import fast
W = mx.contiguous(mx.random.normal((128,64)).astype(mx.float32)); mx.eval(W)
print("compiled:", fast.qwen35_ane_compile_linear(W,1024,1) is not None, flush=True)
