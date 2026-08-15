"""Judge each chosen/rejected pair in an ORPO pairs file with a strong local
model, to catch bad exemplars a keyword filter would miss (e.g. the
motherload failure-loop rows found by manual spot-check).

Uses DeepSeek-V4-Flash-0731-p37-native as judge -- one of the two models the
good-model-bundling harvest was sourced from, so it's judging against its
own demonstrated standard of real narrate+act turns. Plain text model,
loads natively through mlx_lm.

For each row, shows the judge only the CHOSEN target turn (the real
narration + tool call we're about to train the student to imitate) plus a
short summary of the preceding context, and asks for a strict JSON verdict:
  - bundled: does content narrate the SAME action as the tool call (not a
    stalled/looping retry of a previous failed action)?
  - coherent: is the narration coherent, on-task, and not stuck repeating
    itself?
  - keep: overall verdict
  - reason: one sentence

Writes:
  <out>.jsonl       -- all rows with judge verdict attached, sorted worst-first
  <out>.kept.jsonl  -- only rows judged keep=true (this is the file to train on)
  <out>.dropped.jsonl -- rows judged keep=false, for manual review

Usage:
    .venv/bin/python scripts/judge_orpo_pairs.py \
        --pairs data/lora_gemma4/orpo_stopping_pairs_combined.jsonl \
        --model ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-p37-native \
        --out data/lora_gemma4/orpo_stopping_pairs_judged
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlx_vlm import load, generate

JUDGE_SYSTEM = (
    "You are a strict data-quality judge for agentic coding training data. "
    "You will see the tail of a conversation and a candidate assistant turn "
    "that combines narration (content) with a tool call. This turn is meant "
    "to be a POSITIVE example of 'narrate the action, then take it in the "
    "same turn' -- the opposite of stopping after announcing intent.\n\n"
    "Reject the turn if ANY of these apply:\n"
    "- The narration describes being stuck, looping, or repeating a failed "
    "attempt ('I keep doing X', 'stuck in a loop', 'same mistake again', "
    "'broken fragment', 'let me try again' for the Nth time)\n"
    "- The narration doesn't actually match what the tool call does\n"
    "- The narration is empty, generic filler, or incoherent\n"
    "- The tool call arguments look malformed or nonsensical for the stated intent\n\n"
    "Accept the turn if the narration is genuine, on-task, and the tool call "
    "follows naturally from it, even if the narration includes normal "
    "mid-task self-correction (fixing a real bug, adjusting a real plan) -- "
    "that is fine, only REPEATED/LOOPING self-correction is bad.\n\n"
    "Respond with ONLY a JSON object, no other text: "
    '{"bundled": true/false, "coherent": true/false, "keep": true/false, "reason": "<one short sentence>"}'
)


def context_summary(context, max_chars=800):
    lines = []
    for m in context[-6:]:
        role = m["role"]
        text = (m.get("content") or "")[:200]
        if role == "system":
            continue
        lines.append(f"[{role}] {text}")
    s = "\n".join(lines)
    return s[-max_chars:]


def build_prompt(row):
    target = row["chosen"][-1]
    context = row["chosen"][:-1]
    narration = target.get("content") or "(empty)"
    tool_calls = target.get("tool_calls", [])
    tc_summary = json.dumps(
        [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in tool_calls]
    )[:600]
    return (
        f"CONTEXT (last few turns):\n{context_summary(context)}\n\n"
        f"CANDIDATE TURN narration:\n{narration[:800]}\n\n"
        f"CANDIDATE TURN tool call(s):\n{tc_summary}\n\n"
        "Verdict JSON:"
    )


def parse_verdict(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "keep" not in d:
        return None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-p37-native")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    model_path = str(Path(a.model).expanduser())
    print(f"[judge] loading {model_path}", flush=True)
    model, processor = load(model_path)
    tokenizer = getattr(processor, "tokenizer", processor)

    rows = [json.loads(l) for l in open(a.pairs)]
    if a.limit:
        rows = rows[: a.limit]
    print(f"[judge] judging {len(rows)} pairs", flush=True)

    results = []
    for i, row in enumerate(rows):
        prompt = build_prompt(row)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        out = generate(model, processor, text, image=None, max_tokens=a.max_tokens, temperature=0.0, verbose=False)
        out_text = out.text if hasattr(out, "text") else str(out)
        verdict = parse_verdict(out_text)
        if verdict is None:
            verdict = {"bundled": None, "coherent": None, "keep": False, "reason": f"unparseable judge output: {out_text[:200]!r}"}
        results.append({**row, "_judge": verdict})
        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            n_keep = sum(1 for r in results if r["_judge"]["keep"])
            print(f"[judge] {i+1}/{len(rows)} done, {n_keep} kept so far", flush=True)

    results.sort(key=lambda r: (r["_judge"]["keep"], r["_judge"].get("bundled") is not False))

    out_base = a.out
    with open(f"{out_base}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    kept = [r for r in results if r["_judge"]["keep"]]
    dropped = [r for r in results if not r["_judge"]["keep"]]

    with open(f"{out_base}.kept.jsonl", "w") as f:
        for r in kept:
            r2 = {k: v for k, v in r.items() if k != "_judge"}
            f.write(json.dumps(r2) + "\n")

    with open(f"{out_base}.dropped.jsonl", "w") as f:
        for r in dropped:
            f.write(json.dumps(r) + "\n")

    print(f"[judge] total {len(results)}, kept {len(kept)}, dropped {len(dropped)}", flush=True)
    print(f"[judge] wrote {out_base}.jsonl / .kept.jsonl / .dropped.jsonl", flush=True)


if __name__ == "__main__":
    main()
