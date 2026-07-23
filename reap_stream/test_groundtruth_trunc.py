"""Ground-truth test: does head-384 truncation faithfully approximate the
FULL-length saliency (what actually happens at inference)? And does a longer
window approximate it better (validating the fix direction)?

Method: pick affected prompts (ASSISTANT: beyond token 384) that are still
short enough that full-length is memory-tractable (total <= ~1400 tokens).
Run the real saliency collector at three windows on the SAME prompts:
  - 384  (current behavior)
  - 1024 (candidate fix: longer window)
  - full (ground truth, untruncated)
Compare each shorter window against FULL: Jaccard of bottom-25% prune sets
and rank correlation of raw scores. If 1024 matches full much better than
384 does, the fix works. If 384 already matches full, there's no problem.
"""
import json
import numpy as np
from mlx_vlm import load

from .collect_step3p7 import collect_layerwise

MODEL = "models/Step-3.7-Flash"
N = 12
FULL = 4096            # effectively untruncated for our length-filtered set
WINDOWS = [384, 1024, FULL]
MIN_TOTAL, MAX_TOTAL = 500, 1400   # affected but full-length-tractable


def pick_prompts(n):
    _, proc = load(MODEL, lazy=True)
    tok = getattr(proc, "tokenizer", proc)
    out = []
    with open("calib/cloud_reap_8k.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("category") not in ("agentic", "tool_use", "general_instruction"):
                continue
            text = d["text"]
            pos = text.find("ASSISTANT:")
            if pos < 0:
                continue
            total = len(tok.encode(text))
            assistant_at = len(tok.encode(text[:pos]))
            # affected (answer beyond 384) AND full-length tractable
            if assistant_at >= 384 and MIN_TOTAL <= total <= MAX_TOTAL:
                out.append(text)
            if len(out) >= n:
                break
    return out


def _mean_reap(s):
    return np.where(s.reap_count > 0, s.reap_sum / np.maximum(s.reap_count, 1), 0.0)


def prune_sets(stats, frac=0.25):
    out = {}
    for layer, s in stats.items():
        arr = _mean_reap(s)
        k = max(1, int(len(arr) * frac))
        out[layer] = set(np.argsort(arr)[:k].tolist())
    return out


def scores(stats):
    return {layer: _mean_reap(s) for layer, s in stats.items()}


def compare(a_prune, a_scores, b_prune, b_scores, name):
    jaccards, corrs = [], []
    for layer in sorted(set(a_prune) & set(b_prune)):
        pa, pb = a_prune[layer], b_prune[layer]
        jaccards.append(len(pa & pb) / max(1, len(pa | pb)))
        ra = np.argsort(np.argsort(a_scores[layer]))
        rb = np.argsort(np.argsort(b_scores[layer]))
        corrs.append(np.corrcoef(ra, rb)[0, 1])
    print(f"[gt] {name}: MEAN jaccard(prune25%)={np.mean(jaccards):.3f} "
          f"MEAN spearman={np.mean(corrs):.3f}")
    return np.mean(jaccards), np.mean(corrs)


def main():
    prompts = pick_prompts(N)
    print(f"[gt] selected {len(prompts)} affected+tractable prompts")

    results = {}
    for w in WINDOWS:
        label = "full" if w == FULL else str(w)
        print(f"[gt] collecting saliency at window={label}")
        stats, _ = collect_layerwise(
            MODEL, prompts=prompts, max_tokens=w, layers_at_once=6,
            checkpoint_dir=None, resume=False,
        )
        results[w] = (prune_sets(stats), scores(stats))

    fp, fs = results[FULL]
    print("\n[gt] === how well does each window approximate FULL (ground truth)? ===")
    j384, c384 = compare(results[384][0], results[384][1], fp, fs, "384  vs full")
    j1024, c1024 = compare(results[1024][0], results[1024][1], fp, fs, "1024 vs full")

    print("\n[gt] VERDICT:")
    print(f"[gt]   384  -> full : jaccard={j384:.3f} spearman={c384:.3f}")
    print(f"[gt]   1024 -> full : jaccard={j1024:.3f} spearman={c1024:.3f}")
    if j1024 - j384 > 0.08 or c1024 - c384 > 0.05:
        print("[gt]   => longer window matches ground truth MEANINGFULLY better."
              " Fix is justified.")
    elif j384 > 0.85:
        print("[gt]   => 384 already matches ground truth well. No real problem;"
              " re-run not worth it.")
    else:
        print("[gt]   => mixed: 384 imperfect but longer window doesn't clearly"
              " fix it. Needs judgement.")


if __name__ == "__main__":
    main()
