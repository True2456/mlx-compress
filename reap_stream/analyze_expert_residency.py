"""Answer two design questions from an existing REAP telemetry.json.

1. SUBSET RESIDENCY - hold only K of 256 experts in RAM, page the rest from
   SSD. Viability is set by how often a routing decision misses the resident
   set; each miss is an ~9.8 MB read (~1.6 ms).
2. BIT TIERING - hold all experts, spend bits by importance. Viability is set
   by how concentrated importance is, and by whether the ranking metric is
   trustworthy.

Needs no model load: telemetry.json already carries per-layer `freq` (routing
counts) and `reap` (saliency) per expert.

    python -m reap_stream.analyze_expert_residency artifacts/deepseek-v4-reap/telemetry.json
"""
from __future__ import annotations

import argparse, json
import numpy as np

EXPERT_MB = 9.8          # per expert at the current 2/3-bit mix
SSD_GBPS = 6.0
TOP_K = 6


def gini(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("telemetry")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    t = json.load(open(a.telemetry))
    layers = sorted(t, key=int)
    F = np.array([t[l]["freq"] for l in layers], float)
    R = np.array([t[l]["reap"] for l in layers], float)
    L, E = F.shape
    print(f"{L} layers x {E} experts | {int(F.sum())} routing events\n")

    print("=== 1. SUBSET RESIDENCY ===")
    print(f"{'resident':>9} {'slot cov':>9} {'worst lyr':>10} {'miss/token':>11} {'added ms/token':>15}")
    for frac in (0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875):
        k = max(1, int(E * frac))
        cov = np.array([np.sort(F[i])[::-1][:k].sum() / F[i].sum() for i in range(L)])
        miss = a.top_k * L * (1 - cov.mean())
        ms = miss * EXPERT_MB / 1024 / SSD_GBPS * 1000
        print(f"{frac:>8.1%} {cov.mean():>9.1%} {cov.min():>10.1%} {miss:>11.1f} {ms:>15.1f}")

    print("\n=== 2. BIT TIERING ===")
    g = F.sum(0)
    print(f"gini per-layer mean {np.mean([gini(F[i]) for i in range(L)]):.3f}"
          f"   gini global {gini(g):.3f}")
    print("  (per-layer >> global means tiering must be PER LAYER;")
    print("   a single global expert ranking is close to useless here)")
    dead = int((F == 0).sum())
    print(f"experts never routed: {dead}/{F.size} ({dead/F.size:.2%}) -> nothing is free to delete")

    print("\n=== 3. IS `freq` THE RIGHT IMPORTANCE METRIC? ===")
    for frac in (0.25, 0.50, 0.625):
        k = int(E * frac)
        by_f = [np.argsort(-F[i])[:k] for i in range(L)]
        by_r = [np.argsort(-R[i])[:k] for i in range(L)]
        mass_f = np.mean([F[i][by_f[i]].sum() / F[i].sum() for i in range(L)])
        mass_r = np.mean([F[i][by_r[i]].sum() / F[i].sum() for i in range(L)])
        agree = np.mean([len(set(by_f[i].tolist()) & set(by_r[i].tolist())) / k for i in range(L)])
        print(f"  keep {frac:>5.1%}: freq-ranked covers {mass_f:.1%} of mass, "
              f"reap-ranked covers {mass_r:.1%}, sets agree {agree:.1%}")
    print("  REAP's kept set is accuracy-validated; frequency is not.")


if __name__ == "__main__":
    main()
