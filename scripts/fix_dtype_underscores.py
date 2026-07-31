import re

# Known numeric-suffixed dtype stems (MLX + numpy). Auto-repair is safe here
# because each stem+digits has exactly one correct form -- unlike a PID.
_STEMS = r"(bfloat|float|complex|uint|int)"
# stem followed by a digit run that contains at least one delimiter between digits
_CORRUPT = re.compile(_STEMS + r"(\d(?:[ _.,]+\d)+)")

def fix_dtype_underscores(code: str) -> str:
    """Strip delimiters wrongly inserted into dtype numeric suffixes:
    float3_2 -> float32, bfloat1_6 -> bfloat16, complex1_2_8 -> complex128."""
    return _CORRUPT.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), code)

if __name__ == "__main__":
    tests = {
        "mx.bfloat1_6": "mx.bfloat16",
        "x.astype(mx.float3_2)": "x.astype(mx.float32)",
        "dtype=float6_4": "dtype=float64",
        "complex6_4": "complex64",
        "np.complex1_2_8": "np.complex128",      # 3-digit
        "int3_2 and uint8": "int32 and uint8",   # uint8 (no delim) untouched
        "float 3 2 weights": "float32 weights",  # space delimiter
        "a_1 = float32":     "a_1 = float32",    # legit var a_1 untouched, float32 fine
        "score of 9_5":      "score of 9_5",     # NOT a dtype -> left alone (guard flags it)
    }
    ok = 0
    for src, want in tests.items():
        got = fix_dtype_underscores(src)
        mark = "ok " if got == want else "FAIL"
        ok += got == want
        print(f"  {mark} {src!r:26} -> {got!r}")
    print(f"\n{ok}/{len(tests)} passed")
