"""Harvest real bundled-narration+action turns from sessions of models that
don't exhibit the premature-stopping bug, as positive exemplars for
strengthening the stopping-pairs ORPO correction.

Real evidence (checked against actual Herdr orchestration sessions, not
assumed): Step-3.7 produces zero standalone text-only pauses across its
sessions -- it always bundles narration with the tool call in the same
turn, or acts with minimal narration. Gemma-4-12B does not. This harvests
Step-3.7's (and DeepSeek-V4-Flash-0731's) real "narrate + act in one turn"
completions directly from real usage as chosen exemplars -- same
chosen/rejected construction as build_orpo_stopping_data.py (chosen = the
real bundled turn, rejected = same turn with tool_calls stripped), just
sourced from real cross-model evidence instead of derived SFT slices.

Usage:
    .venv/bin/python scripts/harvest_good_model_bundling.py \
        --sessions-dir ~/.pi/agent/sessions \
        --out data/lora_gemma4/orpo_good_model_bundling_pairs.jsonl \
        --max-per-tool 40
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
from collections import Counter, defaultdict

GOOD_MODEL_PATTERNS = ("step-3.7", "deepseek-v4-flash-0731")

SYSTEM_PROMPT = (
    "You are an autonomous coding agent running on the user's machine. "
    "Work in the current directory of a real repository. Use your tools to "
    "read, search, create, and edit files and to run shell commands."
)


def is_good_model(model_id):
    if not model_id:
        return False
    m = model_id.lower()
    return any(p in m for p in GOOD_MODEL_PATTERNS)


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


def load_session(path):
    model = None
    messages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "model_change" and model is None:
                model = d.get("modelId")
            if d.get("type") != "message":
                continue
            m = to_our_message(d["message"])
            if m is not None:
                messages.append(m)
    return model, messages


def find_bundled_turns(messages):
    """Rows where a real assistant turn has both content (narration) and
    tool_calls (action) together -- the behavior we want more of."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls") and (m.get("content") or "").strip():
            if len(m["content"].split()) < 3:
                continue
            context = [{"role": "system", "content": SYSTEM_PROMPT}] + messages[:i]
            out.append((context, m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", default=os.path.expanduser("~/.pi/agent/sessions"))
    ap.add_argument("--out", default="data/lora_gemma4/orpo_good_model_bundling_pairs.jsonl")
    ap.add_argument("--max-per-tool", type=int, default=40)
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.sessions_dir, "**", "*.jsonl"), recursive=True)
    print(f"[bundling] scanning {len(files)} session files", flush=True)

    all_candidates = []
    n_good_sessions = 0
    for fp in files:
        try:
            model, messages = load_session(fp)
        except Exception as e:
            print(f"  skip {fp}: {type(e).__name__}: {e}", flush=True)
            continue
        if not is_good_model(model):
            continue
        n_good_sessions += 1
        all_candidates.extend(find_bundled_turns(messages))

    print(f"[bundling] {n_good_sessions} sessions from good models", flush=True)
    print(f"[bundling] {len(all_candidates)} raw bundled-turn candidates", flush=True)

    per_tool = defaultdict(int)
    selected = []
    for context, target in all_candidates:
        tool_name = target["tool_calls"][0]["function"]["name"]
        if per_tool[tool_name] >= a.max_per_tool:
            continue
        per_tool[tool_name] += 1
        selected.append((context, target, tool_name))

    print(f"[bundling] selected {len(selected)} pairs across {len(per_tool)} distinct tools", flush=True)
    print(f"[bundling] top tools: {Counter(per_tool).most_common(15)}", flush=True)

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
    print(f"[bundling] wrote {len(pairs)} pairs -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
