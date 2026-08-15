"""Small, real, exact-answer eval for the DeepSeek-V4-Flash REAP/REAM/
requantize pipeline -- deliberately not PPL, per this project's own
established lesson (REAM-RESULT.md: a -0.194 NLL "win" bought zero real
accuracy, in some categories negative). Every prompt here has one
objectively checkable answer, so a smoothed/hedged output can't score well
by accident the way it can under perplexity.

No pre-existing DeepSeek-V4 eval set exists in this repo (the Gemma-4/
Step-3.7 ones are model-specific), so this is built fresh: arithmetic
(exact integer), factual recall (short exact string), and small code-output
prediction (exact string, tolerant of code-fence wrapping only). Deliberately
small (24 prompts) given real per-model cost here: each checkpoint load is
~15-30s and this machine is not fast at generation on a model this size, so
the eval must run three full checkpoints without a multi-hour bill.

Usage:
    .venv/bin/python scripts/eval_deepseek_v4_quicktest.py \
        --model models/DeepSeek-V4-Flash-fp8-reap-p15 \
        --label ream_only_p15 \
        --out artifacts/dsv4_eval_ream_only.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from mlx_vlm import generate, load

CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")

PROMPTS = [
    # -- arithmetic, exact integer --
    ("What is 17 * 24? Answer with just the number.", "408"),
    ("What is 156 + 289? Answer with just the number.", "445"),
    ("What is 1000 - 347? Answer with just the number.", "653"),
    ("What is 13 * 13? Answer with just the number.", "169"),
    ("What is 7 * 8 * 2? Answer with just the number.", "112"),
    ("What is 900 / 4? Answer with just the number.", "225"),
    ("What is the remainder when 100 is divided by 7? Answer with just the number.", "2"),
    ("What is 2 to the power of 10? Answer with just the number.", "1024"),
    ("What is 45 * 3 - 15? Answer with just the number.", "120"),
    ("How many minutes are in 4 hours? Answer with just the number.", "240"),
    # -- factual recall, short exact string --
    ("What is the chemical symbol for gold? Answer with just the symbol.", "Au"),
    ("What is the capital of France? Answer with just the city name.", "Paris"),
    ("How many legs does a spider have? Answer with just the number.", "8"),
    ("What planet is closest to the sun? Answer with just the planet name.", "Mercury"),
    ("What is the freezing point of water in Celsius? Answer with just the number.", "0"),
    ("How many sides does a hexagon have? Answer with just the number.", "6"),
    # -- small code-output prediction, exact string (fence-tolerant) --
    ("What does this print?\n```python\nprint(3 + 4 * 2)\n```\nAnswer with just the output.", "11"),
    ("What does this print?\n```python\nprint(len('hello'))\n```\nAnswer with just the output.", "5"),
    ("What does this print?\n```python\nx = [1, 2, 3]\nprint(x[-1])\n```\nAnswer with just the output.", "3"),
    ("What does this print?\n```python\nprint('ab' * 3)\n```\nAnswer with just the output.", "ababab"),
    ("What does this print?\n```python\nprint(10 % 3)\n```\nAnswer with just the output.", "1"),
    ("What does this print?\n```python\nprint(sorted([3, 1, 2]))\n```\nAnswer with just the output.", "[1, 2, 3]"),
    ("What does this print?\n```python\nprint(max(4, 9, 2))\n```\nAnswer with just the output.", "9"),
    ("What does this print?\n```python\nprint(bool(0))\n```\nAnswer with just the output.", "False"),
]


def normalize(text: str) -> str:
    t = text.strip()
    t = CODE_FENCE_RE.sub("", t).strip()
    t = t.strip("`").strip()
    return t


def answers_match(pred: str, expected: str) -> bool:
    """Exact match after normalization, plus a JSON-structural fallback for
    list/dict-shaped answers -- same rationale as eval_gemma4_corrections.py's
    fix earlier this session: plain string equality alone previously
    undercounted a genuinely correct answer over pure formatting (spacing),
    not a hypothetical concern here either given prompt #6 expects '[1, 2, 3]'."""
    p, e = normalize(pred), normalize(expected)
    if p == e:
        return True
    try:
        return json.loads(p) == json.loads(e)
    except Exception:
        return False


def run_eval(model_path: str, max_tokens: int = 24, temperature: float = 0.0, top_p: float = 1.0) -> dict:
    print(f"[eval] loading {model_path} ...", flush=True)
    t0 = time.time()
    model, processor = load(model_path)
    load_s = time.time() - t0
    print(f"[eval] loaded in {load_s:.0f}s", flush=True)
    tok = getattr(processor, "tokenizer", processor)

    results = []
    n_correct = 0
    for i, (prompt, expected) in enumerate(PROMPTS):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        t1 = time.time()
        out = generate(model, processor, rendered, image=None, max_tokens=max_tokens,
                       temperature=temperature, top_p=top_p, verbose=False)
        text = out.text if hasattr(out, "text") else str(out)
        gen_s = time.time() - t1
        pred_norm = normalize(text)
        correct = answers_match(pred_norm, expected)
        n_correct += correct
        results.append({
            "prompt": prompt, "expected": expected, "predicted": pred_norm[:200],
            "correct": correct, "gen_s": round(gen_s, 1),
        })
        print(f"  [{i+1}/{len(PROMPTS)}] {'OK ' if correct else 'FAIL'} "
              f"expected={expected!r} got={pred_norm[:60]!r} ({gen_s:.1f}s)", flush=True)

    acc = n_correct / len(PROMPTS)
    print(f"[eval] {model_path}: {n_correct}/{len(PROMPTS)} = {acc:.3f}", flush=True)
    return {
        "model": model_path, "load_s": round(load_s, 1),
        "n_correct": n_correct, "n_total": len(PROMPTS), "accuracy": acc,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.0,
                     help="0 = greedy (this project's established exact-answer-eval "
                          "practice, see docs/REAM-RESULT.md). DeepSeek-V4-Flash's "
                          "model card recommends 1.0 -- pass that to sanity-check "
                          "greedy isn't degenerate before trusting a greedy comparison.")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--n-prompts", type=int, default=None, help="cap prompt count, for quick spot-checks")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.n_prompts:
        PROMPTS[:] = PROMPTS[:a.n_prompts]
    summary = run_eval(a.model, max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p)
    summary["label"] = a.label or a.model
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(a.out, "w"), indent=2)
        print(f"[eval] wrote -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
