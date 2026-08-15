"""Build ORPO chosen/rejected pairs targeting failure to resume tool-calling
after a text response.

Different transition than scripts/build_orpo_stopping_data.py, which only
covers "narrate intent -> stop before any tool call." Real user report:
model does resume calling tools after being nudged once, but when it
produces a text response mid-task (after some tool calls), it stops the
conversation instead of resuming further tool calls when more are needed.

Checked against the real training data first (not assumed): 15.90% of rows
have their target (the trained completion) be a tool_calls turn that
directly follows a text-only assistant turn in the context -- so this
transition genuinely was in the SFT data, at a meaningful rate, yet the
premature-stop-after-text behavior persists. Same fix shape as before:
build a direct contrastive pair from these specific rows.

chosen = the real row unchanged (context ending in a text-only assistant
turn, target = the real tool_calls turn that follows).
rejected = same row, target's tool_calls stripped (content, if any, kept)
-- i.e. "responds with text/nothing instead of resuming with the tool call."

Usage:
    .venv/bin/python scripts/build_orpo_resume_data.py \
        --data data/lora_gemma4/train.jsonl \
        --out data/lora_gemma4/orpo_resume_pairs.jsonl \
        --target 1000 --max-per-tool 30
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict


def candidate_rows(rows):
    out = []
    for r in rows:
        msgs = r["messages"]
        if len(msgs) < 2:
            continue
        prev, target = msgs[-2], msgs[-1]
        if (
            prev["role"] == "assistant"
            and not prev.get("tool_calls")
            and (prev.get("content") or "").strip()
            and target["role"] == "assistant"
            and target.get("tool_calls")
        ):
            tool_name = target["tool_calls"][0].get("function", {}).get("name", "unknown")
            out.append((r, tool_name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lora_gemma4/train.jsonl")
    ap.add_argument("--out", default="data/lora_gemma4/orpo_resume_pairs.jsonl")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--max-per-tool", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    print(f"[orpo-resume] loaded {len(rows)} rows", flush=True)

    candidates = candidate_rows(rows)
    print(f"[orpo-resume] {len(candidates)} candidates "
          f"(text-only turn directly followed by tool_call target)", flush=True)

    rng = random.Random(a.seed)
    rng.shuffle(candidates)

    per_tool = defaultdict(int)
    selected = []
    for r, tool_name in candidates:
        if per_tool[tool_name] >= a.max_per_tool:
            continue
        per_tool[tool_name] += 1
        selected.append((r, tool_name))
        if len(selected) >= a.target:
            break

    print(f"[orpo-resume] selected {len(selected)} pairs across {len(per_tool)} distinct tools", flush=True)
    print(f"[orpo-resume] top tools: {Counter(per_tool).most_common(10)}", flush=True)

    pairs = []
    for r, _ in selected:
        chosen_messages = r["messages"]
        rejected_messages = copy.deepcopy(r["messages"])
        rejected_messages[-1].pop("tool_calls", None)
        pairs.append({"chosen": chosen_messages, "rejected": rejected_messages})

    with open(a.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[orpo-resume] wrote {len(pairs)} pairs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
