"""Separate text-only (no tool_calls) completion rows from the agentic training set.

Root-cause investigation: the fused gemma-4-12b-frontierdistill model was observed
stopping its turn right after announcing intent in text, instead of following
through to a tool call. Checked against the real training data (not assumed):
30.8% of all 16,730 train.jsonl labels (5,146 rows) are text-only completions
with zero tool calls -- a large, real fraction of the training distribution that
teaches "ending a turn with just text is normal," which plausibly generalizes
into premature stopping. See docs/TOKENIZER-INVESTIGATION.md-style writeups for
the full reasoning.

This does NOT filter out leaked/malformed tool-call syntax -- checked two
candidate rows by hand and both were legitimate: the model had already made a
real structured tool_calls invocation in a prior turn, and the text-only final
turn was a deliberate "print the function-call syntax as text" formatting task,
not a bug.

Usage:
    .venv/bin/python scripts/separate_text_only_completions.py \
        --data data/lora_gemma4/train.jsonl \
        --out-text-only data/lora_gemma4/train_text_only.jsonl \
        --out-filtered data/lora_gemma4/train_no_text_only.jsonl
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lora_gemma4/train.jsonl")
    ap.add_argument("--out-text-only", default="data/lora_gemma4/train_text_only.jsonl")
    ap.add_argument("--out-filtered", default="data/lora_gemma4/train_no_text_only.jsonl")
    ap.add_argument("--max-words", type=int, default=None,
                     help="If set, only rows with completion word count <= this are pulled "
                          "into the text-only file (rest stay in the filtered/action set).")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]

    text_only, filtered = [], []
    for r in rows:
        last = r["messages"][-1]
        content = (last.get("content") or "").strip()
        is_text_only = not last.get("tool_calls") and bool(content)
        if is_text_only and a.max_words is not None:
            is_text_only = len(content.split()) <= a.max_words
        (text_only if is_text_only else filtered).append(r)

    with open(a.out_text_only, "w") as f:
        for r in text_only:
            f.write(json.dumps(r) + "\n")
    with open(a.out_filtered, "w") as f:
        for r in filtered:
            f.write(json.dumps(r) + "\n")

    print(f"[separate] total: {len(rows)}")
    print(f"[separate] text-only -> {a.out_text_only}: {len(text_only)} "
          f"({100*len(text_only)/len(rows):.1f}%)")
    print(f"[separate] filtered (action-taking + rest) -> {a.out_filtered}: {len(filtered)}")


if __name__ == "__main__":
    main()
