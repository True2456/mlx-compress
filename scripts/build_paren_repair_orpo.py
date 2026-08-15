"""Harvest real bracket-imbalance ORPO pairs from DeepSeek-V4 Pi sessions.

Scans ~/.pi/agent/sessions/*.jsonl for toolResult turns carrying a Python
SyntaxError caused by an unbalanced/mismatched bracket (`'(' was never
closed`, `closing parenthesis ')' does not match opening parenthesis '['`,
`unmatched ')'`), while a deepseek-v4* model was the active model at the
time. Confirmed present in real usage (2026-08-06 graphics session, 2026-08-10
Pi-debugging session) -- see docs/DEEPSEEK-V4-FINDINGS.md.

Deliberately NOT mining "chosen" continuations from the log: in the dense
failure-loop case (the 2026-08-10 session), the model never lands cleanly --
it thrashes across bash/write/edit for 6+ turns before escaping, which is
exactly the "motherload failure-loop" pattern judge_orpo_pairs.py already
flags as unfit for positive training data. Instead:

  rejected = the real tool_call exactly as generated (unmodified).
  chosen   = the same tool_call with ONLY the offending line's bracket
             balance corrected (stack-based fix, verified by actually
             compiling the result before being kept -- unverifiable fixes
             are dropped, not guessed at).

This isolates the single behavior being trained (closing-bracket count in
multi-level nesting) instead of a generic "write correct code" target.

Loop dedup: only the first hit per (session, file/line-shape) is kept --
later hits in the same thrash sequence are near-duplicate rows of the same
mistake and would just reinforce over-sampling one moment, not teach the
pattern more broadly.

Usage:
    .venv/bin/python scripts/build_paren_repair_orpo.py \
        --sessions-dir ~/.pi/agent/sessions \
        --out data/lora_gemma4/orpo_paren_repair_pairs.jsonl
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
from collections import Counter

SYSTEM_PROMPT = (
    "You are an autonomous coding agent running on the user's machine. "
    "Work in the current directory of a real repository. Use your tools to "
    "read, search, create, and edit files and to run shell commands."
)

TRACEBACK_LINE_RE = re.compile(
    r'File "[^"]*", line \d+\n(?P<src>.*)\n\s*\^+\n(?P<err>SyntaxError:.*)'
)

OPEN_FOR_CLOSE = {")": "(", "]": "[", "}": "{"}
CLOSE_FOR_OPEN = {v: k for k, v in OPEN_FOR_CLOSE.items()}
CODE_TOOLS = {"bash", "write", "edit"}


def extract_text(content_blocks):
    parts = [c.get("text", "") for c in content_blocks if c.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


def to_our_message(msg):
    role = msg["role"]
    blocks = msg.get("content", [])
    if not isinstance(blocks, list):
        return None
    if role == "user":
        text = extract_text(blocks)
        return {"role": "user", "content": text} if text else None
    if role == "assistant":
        text = extract_text(blocks)
        tool_calls = [
            {
                "id": c.get("id", ""),
                "type": "function",
                "function": {"name": c.get("name", ""), "arguments": c.get("arguments", {})},
            }
            for c in blocks
            if c.get("type") == "toolCall"
        ]
        out = {"role": "assistant", "content": text}
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out
    if role == "toolResult":
        return {"role": "tool", "content": extract_text(blocks)}
    return None


def load_session(path):
    """Returns list of (raw_role, our_message, model_id_at_this_point, raw_toolresult_text)."""
    out = []
    cur_model = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "model_change":
                cur_model = d.get("modelId")
                continue
            if d.get("type") != "message":
                continue
            raw_msg = d["message"]
            m = to_our_message(raw_msg)
            if m is None:
                continue
            raw_text = extract_text(raw_msg.get("content") or []) if raw_msg.get("role") == "toolResult" else None
            out.append((raw_msg["role"], m, cur_model, raw_text))
    return out


def find_bracket_fix(line: str) -> str | None:
    """Stack-based single-line bracket balance fix, ignoring quoted spans.

    On a closer that doesn't match the innermost open bracket, INSERT the
    missing closer(s) for however many levels are actually unclosed and
    keep the original character (it very likely correctly closes an
    ancestor level, not the innermost one) -- rather than replacing the
    character in place. Replace-in-place was the original approach and it's
    wrong whenever the drop happened mid-expression, not at the end: on
    `set(r['type' for r in rows)`, the real bug is one missing `]` after
    `'type'`; the existing `)` was already correctly closing `set(`. The
    old algorithm treated that `)` as "wrong type for the subscript" and
    swapped it, corrupting a correct token to fix an different one -- caught
    via a real elicited generation, verified as producing invalid output
    even after the "fix". Insert-and-recheck handles both that case and the
    multi-level trailing-comment case (`np.array([[...], [...], [...])  #
    comment` needs `]` inserted before the existing `)`, not one `)`
    appended after the comment)."""
    stack = []
    quote = None
    result = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if quote:
            result.append(c)
            if c == "\\" and i + 1 < n:
                result.append(line[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            result.append(c)
        elif c in "([{":
            stack.append(c)
            result.append(c)
        elif c in ")]}":
            while stack and CLOSE_FOR_OPEN[stack[-1]] != c:
                result.append(CLOSE_FOR_OPEN[stack.pop()])
            if stack:
                stack.pop()
                result.append(c)
            # else: extra/unmatched closer with nothing open -> drop it
        else:
            result.append(c)
        i += 1
    if quote is not None:
        return None  # unterminated string, out of scope
    if not stack:
        fixed = "".join(result)
        return fixed if fixed != line else None
    # Still-open brackets at end of line: append closers in LIFO order. A
    # trailing ':' (for/if/def/while/with) must stay last -- appending
    # closers after it produces `...):)`  (confirmed against real corpus
    # lines: `for f in os.listdir(...):`-shaped statements).
    closers = [CLOSE_FOR_OPEN[opener] for opener in reversed(stack)]
    trimmed = "".join(result).rstrip()
    if trimmed.endswith(":"):
        fixed = trimmed[:-1] + "".join(closers) + ":"
    else:
        fixed = "".join(result) + "".join(closers)
    return fixed if fixed != line else None


def _compiles(src: str) -> bool:
    """compile(..., 'exec'), not ast.parse -- ast.parse's grammar accepts
    module-level `return`/`break`/`continue`/`yield` (structurally valid
    AST nodes), which are real SyntaxErrors at compile time. A fixed line
    that only ast.parse-checks clean can still be an invalid standalone
    snippet if the original line lived inside a function/loop body whose
    indentation the traceback doesn't preserve -- compile() catches that,
    ast.parse doesn't."""
    try:
        compile(src, "<verify>", "exec")
        return True
    except SyntaxError:
        return False


def verify_fix(full_text: str, bad_line: str, fixed_line: str) -> str | None:
    """Substitute fixed_line for bad_line in full_text and confirm it parses.
    Falls back to parsing fixed_line alone as a standalone statement."""
    if bad_line in full_text:
        candidate = full_text.replace(bad_line, fixed_line, 1)
        if _compiles(candidate):
            return candidate
    if _compiles(fixed_line.strip()):
        return fixed_line if full_text == bad_line else None
    return None


def find_source_field(tool_call: dict) -> str | None:
    fn = tool_call["function"]
    name, args = fn["name"], fn.get("arguments") or {}
    if name == "bash":
        return args.get("command")
    if name == "write":
        return args.get("content")
    if name == "edit":
        edits = args.get("edits") or []
        return "\n".join(e.get("newText", "") for e in edits)
    return None


def set_source_field(tool_call: dict, old_full: str, new_full: str) -> dict:
    tc = copy.deepcopy(tool_call)
    fn = tc["function"]
    if fn["name"] == "bash":
        fn["arguments"]["command"] = new_full
    elif fn["name"] == "write":
        fn["arguments"]["content"] = new_full
    elif fn["name"] == "edit":
        for e in fn["arguments"].get("edits") or []:
            if e.get("newText", "") in old_full:
                e["newText"] = e["newText"].replace(old_full, new_full) if old_full != e.get("newText", "") else new_full
    return tc


def process_session(path: str, seen_shapes: set) -> list[dict]:
    """Two-tier: prefer relocating the real tool_call that generated the
    broken line (full-file/command context preserved) and correcting only
    that line in place. When relocation fails -- the writing call is out of
    a practical search window, or was itself never logged (e.g. surfaced
    later by a background process, see docs/DEEPSEEK-V4-FINDINGS.md) -- fall
    back to a standalone target built directly from the broken line itself
    (100% real, taken verbatim from the traceback), verified the same way.
    Diagnostic run showed 50/51 distinct broken lines across the corpus fix
    and verify cleanly at the line level alone, so requiring full-context
    relocation was discarding good, real examples rather than protecting
    quality -- this keeps the same verification bar without that bottleneck.
    """
    events = load_session(path)
    pairs = []
    for idx, (role, msg, model, raw_text) in enumerate(events):
        if role != "toolResult" or not raw_text or not model or "deepseek" not in model.lower():
            continue
        if "SyntaxError" not in raw_text:
            continue
        m = TRACEBACK_LINE_RE.search(raw_text)
        if not m:
            continue
        # Deliberately not gating on which SyntaxError phrasing Python chose
        # (was never closed / does not match / unmatched / "perhaps you
        # forgot a comma" / "did you mean 'None'"). Checked the corpus:
        # Python's heuristic message for a missing outer close in a nested
        # call varies by shape (`f(g(x)` vs `f(g(x), h(x)` etc.), but the
        # bracket-fix + compile() gate below is the same regardless -- it's
        # a strictly more accurate filter than matching message text, and
        # relying on it recovered a real second cluster the phrase-matched
        # version was missing entirely.
        bad_line = m.group("src").strip("\n").strip()

        shape_key = (path, bad_line)
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)

        fixed_line = find_bracket_fix(bad_line)
        if fixed_line is None:
            continue

        # Tier 1: relocate the real originating tool_call anywhere earlier
        # in the same session and substitute in place.
        source_call = None
        source_call_idx = None
        for j in range(idx - 1, -1, -1):
            r_j, m_j, _, _ = events[j]
            if r_j != "assistant" or not m_j.get("tool_calls"):
                continue
            for tc in m_j["tool_calls"]:
                if tc["function"]["name"] not in CODE_TOOLS:
                    continue
                field = find_source_field(tc)
                if field and bad_line in field:
                    source_call, source_call_idx = tc, j
                    break
            if source_call:
                break

        if source_call is not None:
            full_text = find_source_field(source_call)
            fixed_full = verify_fix(full_text, bad_line, fixed_line)
            if fixed_full is not None:
                chosen_call = set_source_field(source_call, full_text, fixed_full)
                context = [{"role": "system", "content": SYSTEM_PROMPT}]
                for r, mm, _, _ in events[:source_call_idx]:
                    context.append(mm)
                target_msg = events[source_call_idx][1]
                rejected_target = copy.deepcopy(target_msg)
                chosen_target = copy.deepcopy(target_msg)
                for tc in chosen_target.get("tool_calls", []):
                    if tc["id"] == source_call["id"]:
                        tc["function"]["arguments"] = chosen_call["function"]["arguments"]
                pairs.append({
                    "chosen": context + [chosen_target],
                    "rejected": context + [rejected_target],
                    "meta": {"source_path": path, "bad_line": bad_line,
                             "tool": source_call["function"]["name"], "tier": 1},
                })
                continue

        # Tier 2: fallback -- verify the fix standalone and build a minimal
        # realistic target directly from the real broken line. compile(),
        # not ast.parse: a line needing enclosing-function context (bare
        # return/break/continue/yield) would ast.parse clean but fail to
        # actually run standalone -- see _compiles docstring.
        if not _compiles(fixed_line):
            continue  # genuinely unfixable at line granularity -- drop, don't guess

        context = [{"role": "system", "content": SYSTEM_PROMPT}]
        for r, mm, _, _ in events[:idx]:
            context.append(mm)
        tool_id = f"paren_fix_{len(pairs)}"

        def _target(line):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tool_id, "type": "function",
                    "function": {"name": "bash", "arguments": {"command": f"python3 -c \"{line}\""}},
                }],
            }

        pairs.append({
            "chosen": context + [_target(fixed_line)],
            "rejected": context + [_target(bad_line)],
            "meta": {"source_path": path, "bad_line": bad_line, "tool": "bash", "tier": 2},
        })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", default=os.path.expanduser("~/.pi/agent/sessions"))
    ap.add_argument("--out", default="data/lora_gemma4/orpo_paren_repair_pairs.jsonl")
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.sessions_dir, "**", "*.jsonl"), recursive=True)
    print(f"[paren-repair] scanning {len(files)} session files", flush=True)

    seen_shapes = set()
    all_pairs = []
    per_tool = Counter()
    for fp in files:
        try:
            pairs = process_session(fp, seen_shapes)
        except Exception as e:
            print(f"  skip {fp}: {type(e).__name__}: {e}", flush=True)
            continue
        all_pairs.extend(pairs)
        for p in pairs:
            per_tool[p["meta"]["tool"]] += 1

    per_tier = Counter(p["meta"]["tier"] for p in all_pairs)
    print(f"[paren-repair] {len(all_pairs)} verified pairs across {len(set(p['meta']['source_path'] for p in all_pairs))} sessions", flush=True)
    print(f"[paren-repair] by tool: {dict(per_tool)}  by tier: {dict(per_tier)}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[paren-repair] wrote {len(all_pairs)} pairs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
