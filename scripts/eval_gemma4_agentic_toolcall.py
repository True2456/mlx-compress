"""Tool-call correctness on held-out agentic trajectories: base vs adapted.

HumanEval measures raw code-writing correctness on isolated problems -- a
different distribution than what this LoRA was actually trained for
(multi-turn agentic tool use). This measures the thing the fine-tune was
actually supposed to improve: given real context from a held-out agentic
trajectory (never seen in training), does the model pick the right tool and
fill in reasonable arguments for it, same as the ground-truth continuation
did.

Real, held-out data (data/lora_gemma4/test.jsonl, 720/1016 rows end in an
actual tool call) -- not a synthetic or external benchmark, so it's testing
in-distribution agentic skill directly, complementary to HumanEval's
out-of-distribution code-puzzle check.

Discrete pass/fail grading (tool name match + argument overlap), not
likelihood -- same reasoning as using HumanEval instead of trusting PPL alone.

Usage:
    .venv/bin/python scripts/eval_gemma4_agentic_toolcall.py \
        --model ~/.lmstudio/models/mlx-community/gemma-4-12B-it-qat-4bit \
        --adapter-path adapters/gemma4-12b-agentic \
        --data data/lora_gemma4/test.jsonl --n 150 \
        --out artifacts/toolcall_gemma4_12b_lora.json
"""
from __future__ import annotations

import argparse
import json
import re
import random

from datasets import load_dataset
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

CALL_START_RE = re.compile(r"<\|tool_call>call:(\w+)\{")
STR_ARG_RE = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.DOTALL)
BARE_ARG_RE = re.compile(r"(\w+):([^,}]+)")


def _find_matching_brace(text, start):
    """Scan forward from just after the opening '{', respecting <|"|>...<|"|>
    quoted spans (their contents are opaque to brace counting), to find the
    matching close. Returns (end_index, truncated) -- truncated=True means
    the text ran out before the call closed (max_tokens cutoff), not that the
    call was malformed."""
    depth = 1
    i = start
    n = len(text)
    while i < n:
        if text.startswith('<|"|>', i):
            close = text.find('<|"|>', i + 5)
            if close == -1:
                return n, True
            i = close + 5
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i, False
        i += 1
    return n, True


def parse_tool_calls(text):
    calls = []
    for m in CALL_START_RE.finditer(text):
        name = m.group(1)
        body_start = m.end()
        end, truncated = _find_matching_brace(text, body_start)
        argstr = text[body_start:end]
        args = {}
        consumed = set()
        for k, v in STR_ARG_RE.findall(argstr):
            args[k] = v
            consumed.add(k)
        for k, v in BARE_ARG_RE.findall(argstr):
            if k not in consumed:
                args[k] = v.strip()
        calls.append({"name": name, "arguments": args, "truncated": truncated})
    return calls


def ground_truth_calls(tool_calls):
    out = []
    for tc in tool_calls:
        fn = tc.get("function", tc)
        out.append({"name": fn["name"], "arguments": fn.get("arguments", {})})
    return out


def score_row(pred_calls, gt_calls):
    if not pred_calls or not gt_calls:
        return {"name_match": False, "arg_overlap": 0.0}
    pred0, gt0 = pred_calls[0], gt_calls[0]
    name_match = pred0["name"] == gt0["name"]
    gt_args = gt0["arguments"]
    if not gt_args:
        arg_overlap = 1.0 if name_match else 0.0
    else:
        pred_args = pred0["arguments"]
        n_hit = 0
        for k, v in gt_args.items():
            pv = str(pred_args.get(k, ""))
            if str(v).strip().lower() in pv.lower() or pv.lower() in str(v).strip().lower():
                n_hit += 1
        arg_overlap = n_hit / len(gt_args) if name_match else 0.0
    return {"name_match": name_match, "arg_overlap": arg_overlap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--data", default="data/lora_gemma4/test.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if r["messages"][-1].get("tool_calls")]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    print(f"[eval] {len(rows)} tool-call rows selected", flush=True)

    print(f"[eval] loading {a.model} (adapter={a.adapter_path})", flush=True)
    model, processor = load(
        a.model, adapter_path=a.adapter_path, processor_config={"trust_remote_code": True}
    )
    config = model.config.__dict__

    n_name_match = 0
    arg_overlaps = []
    results = []
    for i, r in enumerate(rows):
        context = r["messages"][:-1]
        gt = ground_truth_calls(r["messages"][-1]["tool_calls"])
        prompt = apply_chat_template(
            processor, config, context, tools=r.get("tools"), add_generation_prompt=True
        )
        out = generate(model, processor, prompt, image=None, max_tokens=a.max_tokens, verbose=False)
        text = out.text if hasattr(out, "text") else str(out)
        pred = parse_tool_calls(text)
        s = score_row(pred, gt)
        n_name_match += s["name_match"]
        arg_overlaps.append(s["arg_overlap"])
        results.append({
            "gt_tool": gt[0]["name"] if gt else None,
            "pred_tool": pred[0]["name"] if pred else None,
            **s,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    summary = {
        "model": a.model,
        "adapter_path": a.adapter_path,
        "n": len(rows),
        "tool_name_accuracy": n_name_match / len(rows),
        "mean_arg_overlap": sum(arg_overlaps) / len(arg_overlaps),
        "results": results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=2)
        print(f"[eval] wrote -> {a.out}")


if __name__ == "__main__":
    main()
