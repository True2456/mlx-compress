"""Unit tests for reap_stream/ream.py on tiny synthetic tensors. No model load.

Run: .venv/bin/python -m reap_stream.test_ream
"""
import numpy as np

from .ream import (assign_merges, merge_experts, merge_layer,
                   router_similarity, build_ream_plan)


def approx(a, b, tol=1e-5):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) < tol


def test_similarity_and_assignment():
    # 4 experts: 0 and 2 point the same way; 1 and 3 the opposite way.
    R = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.02], [0.0, 1.0]], dtype=np.float32)
    sim = router_similarity(R, np)
    assert approx(np.diag(sim), [1, 1, 1, 1]), "self-similarity must be 1"
    # keep {0,1}; prune {2,3}. 2 ~ 0, 3 ~ 1.
    groups = assign_merges([0, 1], [2, 3], sim)
    assert groups == {0: [2], 1: [3]}, groups
    print("ok  similarity + assignment")


def test_merge_is_saliency_weighted():
    # 3 experts, 1-D "weight" each; keep {0}, merge {1,2} into 0.
    W = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    scores = [1.0, 1.0, 2.0]                       # expert 2 weighted 2x
    out = merge_experts(W, keep=[0], groups={0: [1, 2]}, scores=scores, xp=np)
    # (1*10 + 1*20 + 2*30) / (1+1+2) = 90/4 = 22.5
    assert out.shape == (1, 1), out.shape
    assert approx(out, [[22.5]]), out
    print("ok  saliency-weighted merge value")


def test_output_count_equals_keep():
    E, keep = 8, [0, 3, 5]
    W = np.random.randn(E, 4, 6).astype(np.float32)
    groups = {0: [1, 2], 3: [4], 5: [6, 7]}
    out = merge_experts(W, keep, groups, scores=[1.0] * E, xp=np)
    assert out.shape == (len(keep), 4, 6), out.shape
    # a kept expert absorbing nothing is returned unchanged
    lone = merge_experts(W, [3], {}, [1.0] * E, np)
    assert approx(lone[0], W[3]), "kept-only expert must be identity"
    print("ok  output count == len(keep); lone-kept is identity")


def test_zero_saliency_falls_back_to_mean():
    W = np.array([[2.0], [4.0], [6.0]], dtype=np.float32)
    out = merge_experts(W, [0], {0: [1, 2]}, scores=[0.0, 0.0, 0.0], xp=np)
    assert approx(out, [[4.0]]), out          # plain mean (2+4+6)/3
    print("ok  zero-saliency -> plain mean (no div-by-zero)")


def test_merge_preserves_dtype_and_no_mutation():
    W = np.arange(12, dtype=np.float32).reshape(4, 3)
    W_copy = W.copy()
    tensors = {"gate_proj": W, "router_bias": np.arange(4, dtype=np.float32)}
    merged = merge_layer(tensors, [0, 2], {0: [1], 2: [3]}, [1, 1, 1, 1], np)
    assert merged["gate_proj"].shape == (2, 3)
    assert merged["router_bias"].shape == (2,)
    assert merged["gate_proj"].dtype == np.float32
    assert approx(W, W_copy), "inputs must not be mutated"
    print("ok  multi-tensor layer merge; dtype kept; inputs immutable")


def test_plan_is_reap_superset():
    plan = {"method": "reap", "layers": {
        "3": {"num_experts": 4, "keep": [0, 1], "prune": [2, 3],
              "scores": [0.9, 0.8, 0.1, 0.2]}}}
    routers = {3: np.array([[1, 0], [0, 1], [1, 0.02], [0.02, 1]], np.float32)}
    ream = build_ream_plan(plan, routers, np)
    L = ream["layers"]["3"]
    # every REAP field preserved unchanged
    for k in ("keep", "prune", "scores", "num_experts"):
        assert L[k] == plan["layers"]["3"][k], k
    assert "merges" in L and L["merges"], L
    assert ream["method"].startswith("ream("), ream["method"]
    print("ok  REAM plan is a strict superset of the REAP plan")


if __name__ == "__main__":
    for fn in [test_similarity_and_assignment, test_merge_is_saliency_weighted,
               test_output_count_equals_keep, test_zero_saliency_falls_back_to_mean,
               test_merge_preserves_dtype_and_no_mutation, test_plan_is_reap_superset]:
        fn()
    print("\nall REAM unit tests passed")
