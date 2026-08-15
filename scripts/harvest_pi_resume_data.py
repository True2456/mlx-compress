"""Harvest genuine autonomous resume-after-text examples from real Pi session logs.

Real usage data, not synthetic: scans ~/.pi/agent/sessions/*.jsonl for the
exact pattern needed -- an assistant text-only turn (no tool call) directly
followed by another assistant turn that DOES make a tool call, with NO user
message in between. That "no user turn in between" condition is what makes
it a genuine autonomous resume (the model continued on its own), as opposed
to "model stopped, user said proceed, model continued" which is a different,
already-covered case.

chosen = real context (up to and including the text-only turn) + the real
next assistant turn (tool call intact).
rejected = same, with the tool call stripped from the target turn.

Usage:
    .venv/bin/python scripts/harvest_pi_resume_data.py \
        --sessions-dir ~/.pi/agent/sessions \
        --out data/lora_gemma4/orpo_pi_resume_pairs.jsonl \
        --max-per-tool 40
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
from collections import Counter, defaultdict

SYSTEM_PROMPT = (
    "You are an autonomous coding agent running on the user's machine. "
    "Work in the current directory of a real repository. Use your tools to "
    "read, search, create, and edit files and to run shell commands."
)


def extract_text(content_blocks):
    parts = [c.get("text", "") for c in content_blocks if c.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


def to_our_message(msg):
    role = msg["role"]
    content_blocks = msg.get("content", [])
    if not isinstance(content_blocks, list):
        return None
    if role == "user":
        text = extract_text(content_blocks)
        if not text:
            return None
        return {"role": "user", "content": text}
    if role == "assistant":
        text = extract_text(content_blocks)
        tool_calls = []
        for c in content_blocks:
            if c.get("type") == "toolCall":
                tool_calls.append({
                    "id": c.get("id", ""),
                    "type": "function",
                    "function": {"name": c.get("name", ""), "arguments": c.get("arguments", {})},
                })
        out = {"role": "assistant", "content": text}
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out
    if role == "toolResult":
        text = extract_text(content_blocks)
        return {"role": "tool", "content": text}
    return None


def load_session_messages(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "message":
                continue
            m = to_our_message(d["message"])
            if m is not None:
                out.append((d["message"]["role"], m))
    return out


def find_resume_candidates(session_messages):
    """Find (context_messages, target_message) pairs where a text-only
    assistant turn is directly followed by a tool-calling assistant turn
    with no user turn in between."""
    out = []
    raw_roles = [r for r, _ in session_messages]
    msgs = [m for _, m in session_messages]

    i = 0
    while i < len(msgs) - 1:
        role_i, m_i = raw_roles[i], msgs[i]
        if role_i == "assistant" and not m_i.get("tool_calls") and (m_i.get("content") or "").strip():
            # scan forward for the next assistant turn, with no 'user' in between
            j = i + 1
            saw_user = False
            while j < len(msgs):
                if raw_roles[j] == "user":
                    saw_user = True
                    break
                if raw_roles[j] == "assistant":
                    break
                j += 1
            if j < len(msgs) and raw_roles[j] == "assistant" and not saw_user and msgs[j].get("tool_calls"):
                context = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs[: i + 1]
                target = msgs[j]
                out.append((context, target))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", default=os.path.expanduser("~/.pi/agent/sessions"))
    ap.add_argument("--out", default="data/lora_gemma4/orpo_pi_resume_pairs.jsonl")
    ap.add_argument("--max-per-tool", type=int, default=40)
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.sessions_dir, "**", "*.jsonl"), recursive=True)
    print(f"[pi-resume] found {len(files)} session files", flush=True)

    all_candidates = []
    for fp in files:
        try:
            session_messages = load_session_messages(fp)
        except Exception as e:
            print(f"  skip {fp}: {type(e).__name__}: {e}", flush=True)
            continue
        candidates = find_resume_candidates(session_messages)
        all_candidates.extend(candidates)

    print(f"[pi-resume] {len(all_candidates)} raw candidates across all sessions", flush=True)

    per_tool = defaultdict(int)
    selected = []
    for context, target in all_candidates:
        tool_name = target["tool_calls"][0]["function"]["name"]
        if per_tool[tool_name] >= a.max_per_tool:
            continue
        per_tool[tool_name] += 1
        selected.append((context, target, tool_name))

    print(f"[pi-resume] selected {len(selected)} pairs across {len(per_tool)} distinct tools", flush=True)
    print(f"[pi-resume] top tools: {Counter(per_tool).most_common(15)}", flush=True)

    pairs = []
    for context, target, _ in selected:
        chosen_messages = context + [target]
        rejected_target = copy.deepcopy(target)
        rejected_target.pop("tool_calls", None)
        rejected_messages = context + [rejected_target]
        pairs.append({"chosen": chosen_messages, "rejected": rejected_messages})

    out_path = a.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[pi-resume] wrote {len(pairs)} pairs -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
