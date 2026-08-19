"""Capture the real arguments to modelWithMILText:weights:optionsPlist: by
swizzling it from inside the process. No debugger, no entitlements."""
import ctypes, ctypes.util, os, sys

objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
ctypes.cdll.LoadLibrary("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")

objc.objc_getClass.restype = ctypes.c_void_p; objc.objc_getClass.argtypes=[ctypes.c_char_p]
objc.object_getClass.restype = ctypes.c_void_p; objc.object_getClass.argtypes=[ctypes.c_void_p]
objc.sel_registerName.restype = ctypes.c_void_p; objc.sel_registerName.argtypes=[ctypes.c_char_p]
objc.class_getClassMethod.restype = ctypes.c_void_p; objc.class_getClassMethod.argtypes=[ctypes.c_void_p, ctypes.c_void_p]
objc.method_getImplementation.restype = ctypes.c_void_p; objc.method_getImplementation.argtypes=[ctypes.c_void_p]
objc.method_setImplementation.restype = ctypes.c_void_p; objc.method_setImplementation.argtypes=[ctypes.c_void_p, ctypes.c_void_p]
objc.objc_msgSend.restype = ctypes.c_void_p; objc.objc_msgSend.argtypes=[ctypes.c_void_p, ctypes.c_void_p]

def send(obj, name, restype=ctypes.c_void_p):
    f = objc.objc_msgSend
    f.restype = restype; f.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return f(obj, objc.sel_registerName(name.encode()))

def nsdata_bytes(d):
    if not d: return b""
    n = send(d, "length", ctypes.c_ulonglong)
    p = send(d, "bytes", ctypes.c_void_p)
    return ctypes.string_at(p, n) if p and n else b""

def describe(o):
    if not o: return "(nil)"
    s = send(o, "description")
    return nsdata_bytes(send(s, "UTF8String")) if False else ctypes.string_at(
        send(s, "UTF8String", ctypes.c_char_p) or b"")

cls  = objc.objc_getClass(b"_ANEInMemoryModelDescriptor")
meta = objc.object_getClass(cls)
sel  = objc.sel_registerName(b"modelWithMILText:weights:optionsPlist:")
meth = objc.class_getClassMethod(cls, sel)
print("class=%x meta=%x method=%x" % (cls or 0, meta or 0, meth or 0), flush=True)
orig = objc.method_getImplementation(meth)

IMP = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured")
os.makedirs(OUT, exist_ok=True)

@IMP
def hook(self_, cmd, mil, weights, opts):
    try:
        m = nsdata_bytes(mil)
        open(os.path.join(OUT, "mil.txt"), "wb").write(m)
        print("=== CAPTURED MILText: %d bytes ===" % len(m), flush=True)
        print(m.decode("utf-8", "replace")[:4000], flush=True)
        print("=== weights ===", flush=True)
        keys = send(weights, "allKeys")
        n = send(keys, "count", ctypes.c_ulonglong)
        f = objc.objc_msgSend; f.restype=ctypes.c_void_p
        f.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong]
        for i in range(n):
            k = f(keys, objc.sel_registerName(b"objectAtIndex:"), i)
            kc = send(k, "UTF8String", ctypes.c_char_p).decode()
            g = objc.objc_msgSend; g.restype=ctypes.c_void_p
            g.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            v = g(weights, objc.sel_registerName(b"objectForKey:"), k)
            dat = g(v, objc.sel_registerName(b"objectForKey:"),
                    send(objc.objc_getClass(b"NSString"), "string") or 0) if False else None
            ns = objc.objc_getClass(b"NSString")
            h = objc.objc_msgSend; h.restype=ctypes.c_void_p; h.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
            key_data = h(ns, objc.sel_registerName(b"stringWithUTF8String:"), b"data")
            blob = g(v, objc.sel_registerName(b"objectForKey:"), key_data)
            b = nsdata_bytes(blob)
            fn = os.path.join(OUT, kc.split("/")[-1])
            open(fn, "wb").write(b)
            print("  %s -> %s (%d bytes)" % (kc, fn, len(b)), flush=True)
        o = nsdata_bytes(opts)
        open(os.path.join(OUT, "opts.plist"), "wb").write(o)
        print("=== optionsPlist: %d bytes ===" % len(o), flush=True)
        print(describe(opts).decode(errors="replace")[:1500], flush=True)
    except Exception as e:
        print("hook error:", e, flush=True)
    return ctypes.cast(orig, IMP)(self_, cmd, mil, weights, opts)

objc.method_setImplementation(meth, ctypes.cast(hook, ctypes.c_void_p))
print("swizzled", flush=True)

import mlx.core as mx
from omlx.custom_kernels.qwen35_prefill import fast
W = mx.contiguous(mx.random.normal((128, 64)).astype(mx.float32)); mx.eval(W)
m = fast.qwen35_ane_compile_linear(W, 1024, 1)
print("compile done:", m is not None, flush=True)
