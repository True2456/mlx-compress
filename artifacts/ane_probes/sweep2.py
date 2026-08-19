import ctypes, json, sys
exec(open("ane_opsweep.py").read().split("# ---- validate the harness")[0])

MIL = open("captured/mil.txt","rb").read().decode()
WD  = open("captured/weight_data.bin","rb").read()
WS  = open("captured/weight_scale.bin","rb").read()
W = [("@model_path/weights/weight_data.bin", WD),
     ("@model_path/weights/weight_scale.bin", WS)]

print("verbatim captured program:", try_mil(MIL, W))

# substitute: apply a candidate op to the conv output
head, tail = MIL.rsplit("  } -> (y);", 1)
head = head.replace('[name=string("conv")];', '[name=string("conv")];\n@@SLOT@@')
head = head.replace('tensor<fp16, [1, 128, 1, 1024]> y = conv', 'tensor<fp16, [1, 128, 1, 1024]> y0 = conv')
head = head.replace('weight=w, x=x)', 'weight=w, x=x)')
T = "tensor<fp16, [1, 128, 1, 1024]>"

def build(op_line):
    return head.replace("@@SLOT@@", "    " + op_line) + "  } -> (y);" + tail

print("slot sanity (identity):", try_mil(build(f'{T} y = identity(x = y0)[name = string("t")];'), W))

UNARY = ["relu","sigmoid","tanh","silu","gelu","softplus","erf","exp","log","sqrt","rsqrt","abs",
         "sign","floor","ceil","round","square","identity","cos","sin","relu6","logical_not",
         "elu","softmax","l2_norm","cast","clip","leaky_relu","tan","asin","acos","atan",
         "sinh","cosh","atanh","asinh","acosh","threshold","softsign","hard_swish"]
BINARY = ["add","sub","mul","real_div","maximum","minimum","pow","floor_div","mod",
          "greater","less","equal","logical_and","logical_or","matmul","bitwise_and"]

res={"unary":{},"binary":{}}
for op in UNARY:
    ok,why = try_mil(build(f'{T} y = {op}(x = y0)[name = string("t")];'), W)
    res["unary"][op] = True if ok else why
for op in BINARY:
    ok,why = try_mil(build(f'{T} y = {op}(x = y0, y = y0)[name = string("t")];'), W)
    res["binary"][op] = True if ok else why

for kind in ("unary","binary"):
    acc=sorted(k for k,v in res[kind].items() if v is True)
    rej=sorted(k for k,v in res[kind].items() if v is not True)
    print(f"\n{kind} ACCEPTED ({len(acc)}): {', '.join(acc)}")
    print(f"{kind} REJECTED ({len(rej)}): {', '.join(rej)}")
json.dump(res, open("opsweep_result.json","w"), indent=2)
