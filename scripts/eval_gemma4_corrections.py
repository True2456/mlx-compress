"""On-policy failure-correction accuracy: base vs adapted, exact-match.

The three True2456/gemma4-onpolicy-*-corrections datasets are real, mined
Gemma-4-12B failures with exact-match-scorable corrected answers (single
code line, single integer, etc. -- the prompts themselves constrain the
output format). Unlike loss/PPL, this measures whether a change (e.g. the
latent-looping retrofit) actually fixes documented real failures, not just
whether it fits the training distribution better -- the same reasoning that
caught REAM's PPL-over-crediting earlier in this project.

Caveat baked into the data itself, not fixable by this script: these were
mined against gemma-4-12b-it-qat-frontierdistill (the QAT+LoRA-fused
variant), not the clean bf16 base used for the loop retrofit. Some fraction
of "failures" may be specific to that fused model's quirks and may not even
reproduce on the clean base -- run --model against the clean base first as a
sanity baseline before treating a low score as "the retrofit needs work".

Usage:
    .venv/bin/python scripts/eval_gemma4_corrections.py \
        --model ~/.lmstudio/models/mlx-community/gemma-4-12B-it-bf16 \
        --out artifacts/corrections_baseline_bf16.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

DATASETS = [
    "data/eval_corrections/gemma4-onpolicy-50topics-2000-corrections/test.jsonl",
    "data/eval_corrections/gemma4-onpolicy-student-corrections/test.jsonl",
    "data/eval_corrections/gemma4-onpolicy-50topics-corrections/test.jsonl",
]

CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def normalize(text: str) -> str:
    """Strip markdown code fences and surrounding whitespace/quotes -- the
    prompts already constrain format ("return code line only", "return only
    the integer"), so this just tolerates the model wrapping a technically-
    correct answer in fencing it was told not to add."""
    t = text.strip()
    t = CODE_FENCE_RE.sub("", t).strip()
    t = t.strip("`").strip()
    return t


def answers_match(pred: str, expected: str) -> bool:
    """Compare a prediction to the reference answer.

    Plain string equality on normalize() alone measures formatting as much as
    correctness. Measured on the 500-row set, it scored the clean bf16 base at
    9.8% when the base was really at ~17.2%: `student-corrections` expects
    compact JSON ({"x":60,...}) and the model emits conventional spacing
    ({"x": 60, ...}), which is the same answer. That understated baseline then
    masked a genuine 6-point regression from stage-0 training, whose formatting
    gains offset its real losses.

    So JSON is compared STRUCTURALLY. Whitespace is deliberately NOT stripped
    globally -- that would wrongly pass Python answers whose indentation is
    semantic, trading an under-crediting scorer for an over-crediting one (the
    exact failure mode documented in REAM-RESULT.md).
    """
    p, e = normalize(pred), normalize(expected)
    if p == e:
        return True
    try:
        return json.loads(p) == json.loads(e)
    except Exception:
        return False


def load_rows(path: str, n: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    if n is not None:
        rows = rows[:n]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n-per-dataset", type=int, default=None, help="cap rows per dataset, default = all")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"[eval] loading {a.model} (adapter={a.adapter_path})", flush=True)
    model, processor = load(
        a.model, adapter_path=a.adapter_path, processor_config={"trust_remote_code": True}
    )
    config = model.config.__dict__

    per_dataset = {}
    all_results = []
    for ds_path in a.datasets:
        name = Path(ds_path).parent.name
        rows = load_rows(ds_path, a.n_per_dataset)
        n_correct = 0
        for i, r in enumerate(rows):
            msgs = r["messages"]
            user_msg = next(m for m in msgs if m["role"] == "user")
            expected = next(m for m in msgs if m["role"] == "assistant")["content"]

            prompt = apply_chat_template(
                processor, config, [user_msg], add_generation_prompt=True
            )
            out = generate(model, processor, prompt, image=None, max_tokens=a.max_tokens, verbose=False)
            text = out.text if hasattr(out, "text") else str(out)

            pred_norm = normalize(text)
            exp_norm = normalize(expected)
            correct = pred_norm == exp_norm
            n_correct += correct
            all_results.append({
                "dataset": name,
                "prompt": user_msg["content"][:200],
                "expected": exp_norm,
                "predicted": pred_norm[:300],
                "correct": correct,
            })
            if (i + 1) % 25 == 0:
                print(f"  [{name}] {i+1}/{len(rows)} (acc so far: {n_correct/(i+1):.3f})", flush=True)

        acc = n_correct / len(rows) if rows else 0.0
        per_dataset[name] = {"n": len(rows), "accuracy": acc}
        print(f"[eval] {name}: {n_correct}/{len(rows)} = {acc:.3f}", flush=True)

    total_n = sum(d["n"] for d in per_dataset.values())
    total_correct = sum(d["n"] * d["accuracy"] for d in per_dataset.values())
    overall_acc = total_correct / total_n if total_n else 0.0

    summary = {
        "model": a.model,
        "adapter_path": a.adapter_path,
        "per_dataset": per_dataset,
        "overall_accuracy": overall_acc,
        "overall_n": total_n,
        "results": all_results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(a.out, "w"), indent=2)
        print(f"[eval] wrote -> {a.out}")


if __name__ == "__main__":
    main()
