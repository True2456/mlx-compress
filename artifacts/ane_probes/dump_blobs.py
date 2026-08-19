import ctypes, os
objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
ctypes.cdll.LoadLibrary("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")
objc.objc_getClass.restype=ctypes.c_void_p; objc.objc_getClass.argtypes=[ctypes.c_char_p]
objc.object_getClass.restype=ctypes.c_void_p; objc.object_getClass.argtypes=[ctypes.c_void_p]
objc.sel_registerName.restype=ctypes.c_void_p; objc.sel_registerName.argtypes=[ctypes.c_char_p]
objc.class_getClassMethod.restype=ctypes.c_void_p; objc.class_getClassMethod.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
objc.method_getImplementation.restype=ctypes.c_void_p; objc.method_getImplementation.argtypes=[ctypes.c_void_p]
objc.method_setImplementation.restype=ctypes.c_void_p; objc.method_setImplementation.argtypes=[ctypes.c_void_p,ctypes.c_void_p]

def S(n): return objc.sel_registerName(n.encode())
def m0(o,s,rt=ctypes.c_void_p):
    f=objc.objc_msgSend; f.restype=rt; f.argtypes=[ctypes.c_void_p]*2; return f(o,S(s))
def m1(o,s,a,rt=ctypes.c_void_p,at=ctypes.c_void_p):
    f=objc.objc_msgSend; f.restype=rt; f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,at]; return f(o,S(s),a)
def nsstr(t):
    f=objc.objc_msgSend; f.restype=ctypes.c_void_p
    f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_char_p]
    return f(objc.objc_getClass(b"NSString"),S("stringWithUTF8String:"),t.encode())
def raw(d):
    if not d: return b""
    n=m0(d,"length",ctypes.c_ulonglong); p=m0(d,"bytes",ctypes.c_void_p)
    return ctypes.string_at(p,n) if p and n else b""
def utf8(s):
    cp=m0(s,"UTF8String",ctypes.c_char_p); return cp.decode() if cp else ""

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"captured"); os.makedirs(OUT,exist_ok=True)
cls=objc.objc_getClass(b"_ANEInMemoryModelDescriptor")
meth=objc.class_getClassMethod(cls,S("modelWithMILText:weights:optionsPlist:"))
orig=objc.method_getImplementation(meth)
IMP=ctypes.CFUNCTYPE(*([ctypes.c_void_p]*6))
@IMP
def hook(self_,cmd,mil,weights,opts):
    try:
        open(os.path.join(OUT,"mil.txt"),"wb").write(raw(mil))
        keys=m0(weights,"allKeys"); n=m0(keys,"count",ctypes.c_ulonglong)
        for i in range(n):
            f=objc.objc_msgSend; f.restype=ctypes.c_void_p
            f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulonglong]
            k=f(keys,S("objectAtIndex:"),i)
            name=utf8(k)
            v=m1(weights,"objectForKey:",k)
            blob=m1(v,"objectForKey:",nsstr("data"))
            off=m1(v,"objectForKey:",nsstr("offset"))
            b=raw(blob)
            fn=os.path.join(OUT,name.split("/")[-1])
            open(fn,"wb").write(b)
            print(f"  {name} -> {len(b)} bytes, offset={m0(off,'longLongValue',ctypes.c_longlong)}",flush=True)
            print(f"    header[0:64]={b[:64].hex()}",flush=True)
    except Exception as e:
        print("hook err:",e,flush=True)
    return ctypes.cast(orig,IMP)(self_,cmd,mil,weights,opts)
objc.method_setImplementation(meth,ctypes.cast(hook,ctypes.c_void_p))

import mlx.core as mx
from omlx.custom_kernels.qwen35_prefill import fast
W=mx.contiguous(mx.random.normal((128,64)).astype(mx.float32)); mx.eval(W)
print("compiled:",fast.qwen35_ane_compile_linear(W,1024,1) is not None,flush=True)
