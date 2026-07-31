#!/usr/bin/env python3
"""Catch Step-3.7's numeric corruption before a tool call executes.

Meant to be copied into an agent harness as a pre-tool-use check, not run as
part of the REAP pipeline. See the HF README's "numbers corrupted inside the
reasoning block" section for the measured behaviour this defends against.

Why code and not the model: the failure survives self-review. In one traced
run the model re-read the question 20+ times and reaffirmed the wrong value
each time, because by then its own output was the strongest evidence in
context. A checker that does not share that context does not share the bias.

Three checks, cheapest first:

  scan_signature(text)      known corruption shapes -- 4.4.7, "1 8 4 5",
                            2,4,5,6,1,3,3,7. No source needed.
  check_provenance(cmd,src) every numeral in cmd must appear in src. Catches
                            invented numbers that happen to look well-formed.
  check_sed_bounds(cmd)     sed line ranges against the real file length.

Usage:
    from numeric_guard import guard
    problems = guard(proposed_command, source_text=file_contents)
    if problems: ...retry instead of executing...

    $ python3 scripts/numeric_guard.py "sed -n '4.4.7,4.9.5p' src/aero.py"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# 1.2.0 / 4.4.7 -- a numeral carrying more interior dots than a decimal can.
# Version-like strings are legitimate in many contexts, so this is reported
# rather than treated as certainly wrong; see guard()'s `strict` flag.
MULTI_DOT = re.compile(r"(?<![\w.])\d+\.\d+\.\d+(?![\w.])")

# "1 8 4 5" / "2 4 5 6" -- three or more single digits separated by spaces.
SPACED_DIGITS = re.compile(r"(?<!\d)\d(?: \d){2,}(?!\d)")

# "2,4,5,6,1,3,3,7" -- single digits comma-separated. Distinguished from a
# real list like "2456, 1337" by every element being exactly one digit.
COMMA_DIGITS = re.compile(r"(?<!\d)\d(?:,\d){3,}(?!\d)")

NUMERALS = re.compile(r"\d+")
SED_RANGE = re.compile(r"sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+(\S+)")

# 5_1_8_4_0_0 -- underscore inserted between single digits. Python's numeric
# separators make `_` a legitimate delimiter, which primes exactly this. Note
# `5_1_8_4_0_0.0 == 518400.0` is True, so in Python this is cosmetic rather
# than wrong -- but it is a hard error in JSON, YAML and shell.
# Trailing `.` must be allowed -- these appear as float literals (5_1_8_4_0_0.0).
UNDERSCORE_NUM = re.compile(r"(?<![\w.])\d[\d_]*_[\d_]*\d(?!\w)")


def _bad_underscore_grouping(lit: str) -> bool:
    """True unless the literal uses conventional 3-digit grouping.

    Legitimate: 5_000_000, 1_234, 12_345_678 -- every group after the first is
    exactly 3 digits and the first is 1-3. Anything else (5_1_8_4, 1_00_000)
    is the corruption signature.
    """
    groups = lit.split("_")
    if any(g == "" for g in groups):
        return True
    return not (1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]))


def scan_signature(text: str, strict: bool = False) -> list[str]:
    """Known corruption shapes. No source text required."""
    out = []
    for m in SPACED_DIGITS.finditer(text):
        out.append(f"digits split by spaces: {m.group(0)!r}")
    for m in COMMA_DIGITS.finditer(text):
        out.append(f"digits split by commas: {m.group(0)!r}")
    for m in UNDERSCORE_NUM.finditer(text):
        if _bad_underscore_grouping(m.group(0)):
            out.append(f"digits split by underscores: {m.group(0)!r} "
                       f"(valid Python, but wrong in JSON/YAML/shell)")
    for m in MULTI_DOT.finditer(text):
        label = "malformed number" if strict else "version-like numeral (check)"
        out.append(f"{label}: {m.group(0)!r}")
    return out


def check_provenance(command: str, source: str, min_len: int = 3) -> list[str]:
    """Every numeral of >=min_len digits in `command` must occur in `source`.

    min_len avoids flagging small incidental numbers (-9, exit codes, 0/1).
    Numbers the model legitimately *computed* will also trip this, so treat
    hits as "confirm before running", not as proof of corruption.
    """
    return [f"numeral {n!r} does not appear in the source"
            for n in {m.group(0) for m in NUMERALS.finditer(command)}
            if len(n) >= min_len and n not in source]


def check_sed_bounds(command: str, root: str | Path = ".") -> list[str]:
    """sed line ranges against the file's real length."""
    out = []
    for start, end, path in SED_RANGE.findall(command):
        p = Path(root) / path
        if not p.exists():
            continue
        n = sum(1 for _ in p.open(errors="replace"))
        s, e = int(start), int(end)
        if s > e:
            out.append(f"sed range {s},{e} is inverted")
        if s > n or e > n:
            out.append(f"sed range {s},{e} exceeds {path} ({n} lines)")
    return out


def check_python_int_positions(code: str) -> list[str]:
    """Float literals where Python requires an int: slice indices, range().

    This is the corruption's most dangerous form, because one inserted '.'
    yields a *valid* float that no shape-based check can distinguish from a
    legitimate one. `content[idx-5:idx+1.5]` parses fine and fails only at
    runtime with "slice indices must be integers" -- the exact loop seen in
    the wild. Catching it needs the syntactic position, not the literal.
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    out = []

    def floats_in(node):
        return [n for n in ast.walk(node)
                if isinstance(n, ast.Constant) and isinstance(n.value, float)]

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            for f in floats_in(node.slice):
                out.append(f"float {f.value!r} used as a slice index "
                           f"(line {f.lineno}) -- fails at runtime")
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "range":
            for arg in node.args:
                for f in floats_in(arg):
                    out.append(f"float {f.value!r} passed to range() "
                               f"(line {f.lineno}) -- fails at runtime")
    return out


def guard(command: str, source_text: str | None = None,
          root: str | Path = ".", strict: bool = False,
          as_python: bool = False) -> list[str]:
    """All applicable checks. Empty list means nothing suspicious."""
    problems = scan_signature(command, strict=strict)
    if source_text is not None:
        problems += check_provenance(command, source_text)
    problems += check_sed_bounds(command, root)
    if as_python:
        problems += check_python_int_positions(command)
    return problems


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip().split("Usage:")[-1].strip())
        raise SystemExit(2)
    cmd = sys.argv[1]
    src = Path(sys.argv[2]).read_text() if len(sys.argv) > 2 else None
    found = guard(cmd, source_text=src)
    if not found:
        print("ok")
    else:
        print(f"SUSPECT: {cmd!r}")
        for p in found:
            print(f"  - {p}")
        raise SystemExit(1)
