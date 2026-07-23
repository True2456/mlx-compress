"""Which 1024-token truncation mode best approximates FULL-length saliency,
on prompts where truncation actually happens (>1024 total tokens)?

Compares head-1024 / tail-1024 / headtail-1024 against full (ground truth)
on prompts with total tokens in [1200, 2400] -- long enough that 1024 drops
real content, short enough that full-length is memory-tractable. The winner
is the mode we use for the real 5k re-run.
"""
import json
import numpy as np
from mlx_vlm import load

from .collect_step3p7 import collect_layerwise

MODEL = "models/Step-3.7-Flash"
N = 12
WIN = 1024
FULL = 4096
MIN_TOTAL, MAX_TOTAL = 1200, 2400


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
            total = len(tok.encode(text))
            if MIN_TOTAL <= total <= MAX_TOTAL:
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


def compare(a, b, name):
    ap, as_ = a
    bp, bs = b
    jacc, corr = [], []
    for layer in sorted(set(ap) & set(bp)):
        jacc.append(len(ap[layer] & bp[layer]) / max(1, len(ap[layer] | bp[layer])))
        ra = np.argsort(np.argsort(as_[layer]))
        rb = np.argsort(np.argsort(bs[layer]))
        corr.append(np.corrcoef(ra, rb)[0, 1])
    j, c = np.mean(jacc), np.mean(corr)
    print(f"[modes] {name}: jaccard={j:.3f} spearman={c:.3f}")
    return j, c


def main():
    prompts = pick_prompts(N)
    print(f"[modes] {len(prompts)} prompts, total tokens in [{MIN_TOTAL},{MAX_TOTAL}]")

    def run(win, mode):
        print(f"[modes] collecting win={win} mode={mode}")
        stats, _ = collect_layerwise(MODEL, prompts=prompts, max_tokens=win,
                                     layers_at_once=6, checkpoint_dir=None,
                                     resume=False, truncation=mode)
        return prune_sets(stats), scores(stats)

    full = run(FULL, "head")   # full length: mode irrelevant, nothing truncated
    head = run(WIN, "head")
    tail = run(WIN, "tail")
    headtail = run(WIN, "headtail")

    print("\n[modes] === each 1024 mode vs FULL ground truth ===")
    jh, ch = compare(head, full, "head-1024     vs full")
    jt, ct = compare(tail, full, "tail-1024     vs full")
    jht, cht = compare(headtail, full, "headtail-1024 vs full")

    ranked = sorted([("head", jh, ch), ("tail", jt, ct), ("headtail", jht, cht)],
                    key=lambda x: (x[1] + x[2]), reverse=True)
    print(f"\n[modes] BEST approximation of ground truth: {ranked[0][0]} "
          f"(jaccard={ranked[0][1]:.3f} spearman={ranked[0][2]:.3f})")


if __name__ == "__main__":
    main()
