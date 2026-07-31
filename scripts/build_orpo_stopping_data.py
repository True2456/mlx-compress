"""Build ORPO chosen/rejected pairs targeting premature turn-stopping.

Root cause (see memory: agentic-data-text-only-completion-rate): the fused
gemma-4-12b-frontierdistill model stops its turn right after announcing intent
in text instead of following through to the tool call. 30.8% of the original
16,730 SFT labels are text-only (no tool_calls), teaching a base rate that
generalizes too aggressively into premature stopping.

This builds preference pairs from real ground-truth trajectories, no new data
collection needed: for rows where the true completion narrates before acting
(content + tool_calls both present), chosen = the real turn unchanged,
rejected = the identical turn with tool_calls dropped -- i.e. "announced
intent, then just stopped," reproducing the exact observed bug shape.

Sampled for tool-name diversity (cap per tool) rather than dumping the whole
~11.5k action-taking pool -- this is a narrow behavioral correction, not a
skill to teach, and greghavens' own precedent + our on-policy correction sets
both land in the hundreds-to-low-thousands range for targeted fixes, not 10k+.

Usage:
    .venv/bin/python scripts/build_orpo_stopping_data.py \
        --data data/lora_gemma4/train_no_text_only.jsonl \
        --out data/lora_gemma4/orpo_stopping_pairs.jsonl \
        --target 1000 --max-per-tool 30 --min-words 3
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict


def candidate_rows(rows, min_words):
    out = []
    for r in rows:
        last = r["messages"][-1]
        content = (last.get("content") or "").strip()
        tool_calls = last.get("tool_calls")
        if content and tool_calls and len(content.split()) >= min_words:
            tool_name = tool_calls[0].get("function", {}).get("name", "unknown")
            out.append((r, tool_name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lora_gemma4/train_no_text_only.jsonl")
    ap.add_argument("--out", default="data/lora_gemma4/orpo_stopping_pairs.jsonl")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--max-per-tool", type=int, default=30)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    print(f"[orpo] loaded {len(rows)} rows", flush=True)

    candidates = candidate_rows(rows, a.min_words)
    print(f"[orpo] {len(candidates)} candidates (content + tool_calls, >= {a.min_words} words)", flush=True)

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

    print(f"[orpo] selected {len(selected)} pairs across {len(per_tool)} distinct tools", flush=True)
    top_tools = Counter({k: v for k, v in per_tool.items()}).most_common(10)
    print(f"[orpo] top tools: {top_tools}", flush=True)

    pairs = []
    for r, _ in selected:
        chosen_messages = r["messages"]
        rejected_messages = copy.deepcopy(r["messages"])
        rejected_messages[-1].pop("tool_calls", None)
        pairs.append({"chosen": chosen_messages, "rejected": rejected_messages})

    with open(a.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[orpo] wrote {len(pairs)} pairs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
