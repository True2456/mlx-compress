"""Empirical test: does truncating to the first 384 tokens (current behavior,
cutting off before ASSISTANT: for most agentic/tool_use prompts) actually
produce different REAP saliency than a version that includes the real answer?

Method: pick N prompts from the affected categories where ASSISTANT: currently
falls beyond token 384. Build two variants:
  - head: current behavior (first 384 tokens -- setup text only)
  - tail: last 384 tokens of the SAME original text (guarantees the answer
    is included)
Run the real layerwise saliency collector on both, then compare per-layer
saliency rankings (Jaccard overlap of bottom-25% "prune" sets, correlation of
raw scores). Low overlap / low correlation = the truncation bug matters.
High overlap = it doesn't matter much in practice.
"""
import json
import numpy as np
from mlx_vlm import load

from .collect_step3p7 import collect_layerwise

MODEL = "models/Step-3.7-Flash"
N = 30
MAX_TOKENS = 384


def pick_affected_prompts(n):
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
            n_tok_before = len(tok.encode(text[:pos]))
            if n_tok_before < MAX_TOKENS:
                continue  # not actually affected, skip
            out.append((text, tok))
            if len(out) >= n:
                break
    return out, tok


def build_tail_text(text, tok, max_tokens):
    ids = tok.encode(text)
    tail_ids = ids[-max_tokens:]
    return tok.decode(tail_ids)


def _mean_reap(s):
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(s.reap_count > 0, s.reap_sum / np.maximum(s.reap_count, 1), 0.0)
    return m


def saliency_prune_sets(stats, frac=0.25):
    """Return {layer: set(bottom-frac expert ids)} from collector stats."""
    out = {}
    for layer, s in stats.items():
        arr = _mean_reap(s)
        n_prune = max(1, int(len(arr) * frac))
        order = np.argsort(arr)  # ascending: lowest saliency first
        out[layer] = set(order[:n_prune].tolist())
    return out


def raw_scores(stats):
    return {layer: _mean_reap(s) for layer, s in stats.items()}


def main():
    print(f"[test] selecting {N} truncation-affected prompts")
    picked, tok = pick_affected_prompts(N)
    print(f"[test] found {len(picked)} affected prompts")

    head_prompts = [text for text, _ in picked]
    tail_prompts = [build_tail_text(text, tok, MAX_TOKENS) for text, _ in picked]

    print("[test] collecting saliency: HEAD (current behavior, setup-text only)")
    stats_head, _ = collect_layerwise(
        MODEL, prompts=head_prompts, max_tokens=MAX_TOKENS, layers_at_once=6,
        checkpoint_dir=None, resume=False,
    )
    print("[test] collecting saliency: TAIL (guarantees answer included)")
    stats_tail, _ = collect_layerwise(
        MODEL, prompts=tail_prompts, max_tokens=MAX_TOKENS, layers_at_once=6,
        checkpoint_dir=None, resume=False,
    )

    prune_head = saliency_prune_sets(stats_head)
    prune_tail = saliency_prune_sets(stats_tail)
    scores_head = raw_scores(stats_head)
    scores_tail = raw_scores(stats_tail)

    print("\n[test] per-layer comparison (head vs tail):")
    print(f"{'layer':>5} {'jaccard(prune25%)':>18} {'spearman(scores)':>18}")
    jaccards, corrs = [], []
    for layer in sorted(set(prune_head) & set(prune_tail)):
        a, b = prune_head[layer], prune_tail[layer]
        jac = len(a & b) / max(1, len(a | b))
        sh, st = scores_head[layer], scores_tail[layer]
        # rank correlation
        rank_h = np.argsort(np.argsort(sh))
        rank_t = np.argsort(np.argsort(st))
        corr = np.corrcoef(rank_h, rank_t)[0, 1]
        jaccards.append(jac)
        corrs.append(corr)
        print(f"{layer:>5} {jac:>18.3f} {corr:>18.3f}")

    print(f"\n[test] MEAN jaccard(prune25%%)={np.mean(jaccards):.3f} "
          f"MEAN spearman={np.mean(corrs):.3f}")
    print("[test] (jaccard/corr near 1.0 = truncation barely matters; "
          "near 0 = it matters a lot)")


if __name__ == "__main__":
    main()
