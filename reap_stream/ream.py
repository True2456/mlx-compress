"""REAM: merge low-saliency experts into their nearest kept neighbour, instead
of hard-pruning them (as REAP does).

Additive and self-contained. Nothing here replaces or imports the existing
prune path -- `build_student*.py` and `apply_step3p7.py` are untouched. A REAM
plan is a strict superset of a REAP prune plan (same `keep`/`prune`/`scores`,
plus a `merges` block), so any REAP consumer keeps working on a REAM plan and
simply ignores `merges`.

WHY this model, per docs/FINDINGS.md: Step-3.7 is prune-resistant (flat
saliency, zero dead experts, every expert fires), so pruning discards real,
non-redundant capacity. Merging folds some of that capacity back into a kept
expert rather than deleting it. REAM also reuses the existing REAP saliency
scores directly -- no new collection pass to *build* a plan.

MERGE SEMANTICS (stated, not assumed). For a kept expert k that absorbs the
set D_k of pruned experts assigned to it, each weight matrix becomes the
saliency-weighted average

    W_k' = ( s_k*W_k + sum_{d in D_k} s_d*W_d ) / ( s_k + sum_{d in D_k} s_d )

applied independently to gate_proj, up_proj, down_proj, the router row
gate.gate[k], and router_bias[k]. Output dimension is exactly n_keep, identical
to pruning -- REAM changes the *values* of the kept experts, not the count.

HONEST LIMITATION. SwiGLU is nonlinear, so weight-space averaging is an
*approximation* of merging two experts' functions -- exact only for the linear
parts (it is a genuine function-merge for down_proj, approximate for the gated
gate/up pair). This is standard for expert-merging methods and is exactly why a
REAM build must be validated empirically (500-prompt PPL + 250-image NLL vs the
current deploy model), never assumed to help.

ASSIGNMENT. Each pruned expert is merged into the most *similar* kept expert.
Similarity ideally comes from co-activation (docs/FINDINGS.md "cheapest decisive
experiment": if the off-diagonal co-occurrence C_{i,j} ~= 0 the experts are
orthogonal and merging cannot help -- that gate is not yet measured). Until a
co-occurrence matrix exists, `router_similarity` uses router-row cosine as the
stand-in: experts whose router directions align tend to fire on similar tokens.
`assign_merges` accepts any similarity matrix, so a real co-occurrence matrix
drops in unchanged.
"""
from __future__ import annotations

from typing import Sequence


def _asarray(xp, data, dtype):
    """numpy has asarray; mlx.core has array. Support both."""
    fn = getattr(xp, "asarray", None) or xp.array
    return fn(data, dtype=dtype)


def router_similarity(router_rows, xp):
    """Cosine similarity between router rows. router_rows: [E, H] (BF16 base
    weights, NOT quantized). Returns [E, E]. `xp` is numpy or mlx.core."""
    R = router_rows.astype(xp.float32) if hasattr(router_rows, "astype") else router_rows
    norm = xp.sqrt(xp.sum(R * R, axis=1, keepdims=True)) + 1e-9
    Rn = R / norm
    return Rn @ Rn.T


def assign_merges(keep: Sequence[int], prune: Sequence[int], sim) -> dict[int, list[int]]:
    """Assign each pruned expert to the kept expert it is most similar to.

    Returns {kept_expert_id: [pruned ids merged into it]}. Kept experts that
    absorb nothing are omitted. `sim` is an [E, E] similarity matrix
    (array-like, indexable as sim[p][k])."""
    keep = list(keep)
    groups: dict[int, list[int]] = {}
    for p in prune:
        best_k, best_s = keep[0], float("-inf")
        row = sim[p]
        for k in keep:
            s = float(row[k])
            if s > best_s:
                best_s, best_k = s, k
        groups.setdefault(best_k, []).append(p)
    return groups


def build_ream_plan(prune_plan: dict, router_rows_by_layer: dict, xp) -> dict:
    """Turn a REAP prune plan into a REAM merge plan (superset).

    prune_plan: the existing plan dict (plan_p15_blend03.json shape).
    router_rows_by_layer: {layer_int: router_rows [E,H]} BF16, for similarity.
    Adds, per layer, a `merges` dict {kept_id: [pruned ids]}. `keep`/`prune`/
    `scores` are left exactly as-is so REAP consumers are unaffected."""
    out = dict(prune_plan)
    out["method"] = "ream(" + str(prune_plan.get("method", "reap")) + ")"
    out["ream_note"] = ("merges pruned experts into nearest kept by router-row "
                        "cosine; replace with co-occurrence when available")
    layers = {}
    for lk, lp in prune_plan["layers"].items():
        li = int(lk)
        entry = dict(lp)
        if li in router_rows_by_layer:
            sim = router_similarity(router_rows_by_layer[li], xp)
            groups = assign_merges(lp["keep"], lp["prune"], sim)
            entry["merges"] = {str(k): v for k, v in groups.items()}
        layers[lk] = entry
    out["layers"] = layers
    return out


def merge_experts(W, keep: Sequence[int], groups: dict[int, list[int]],
                  scores: Sequence[float], xp):
    """Saliency-weighted merge of a stacked expert tensor.

    W: [E, ...] stacked over experts (gate_proj/up_proj/down_proj, or a router
    matrix [E, H], or a router bias [E]). keep: ordered kept ids. groups:
    {kept_id: [absorbed pruned ids]}. scores: full length-E saliency vector.
    Returns [n_keep, ...] in the order of `keep`, so it is a drop-in for the
    prune path's `W[keep]`."""
    rows = []
    for k in keep:
        ids = [k] + groups.get(k, [])
        w = _asarray(xp, [max(float(scores[i]), 0.0) for i in ids], xp.float32)
        total = float(w.sum())
        if total <= 1e-12:                      # all-zero saliency: plain mean
            w = _asarray(xp, [1.0] * len(ids), xp.float32)
            total = float(len(ids))
        idx = _asarray(xp, ids, xp.int32)
        stack = xp.take(W, idx, axis=0).astype(xp.float32)   # [len(ids), ...]
        shape = [len(ids)] + [1] * (stack.ndim - 1)
        merged = xp.sum(stack * w.reshape(shape), axis=0) / total
        rows.append(merged.astype(W.dtype))
    return xp.stack(rows, axis=0)


def merge_layer(tensors: dict, keep: Sequence[int], groups: dict[int, list[int]],
                scores: Sequence[float], xp) -> dict:
    """Apply merge_experts to every per-expert tensor of one MoE layer.

    tensors: {name: stacked array}, names among gate_proj/up_proj/down_proj/
    router_weight/router_bias. Returns the merged (reduced) tensors in the same
    dict shape. Pure -- does not mutate the inputs."""
    return {name: merge_experts(W, keep, groups, scores, xp)
            for name, W in tensors.items()}
