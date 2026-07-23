"""Are vision experts SEPARATE from text experts, or MIXED?

Compares the vision-only saliency map against the text-only one and reports:
  1. Rank correlation per layer (high => same experts serve both modalities)
  2. Overlap of each modality's TOP-25% experts
  3. THE KEY NUMBER: how many of vision's top experts fall inside the text-only
     plan's PRUNE set -- i.e. how much vision capability the current plan deletes.

Verdict:
  MIXED    -> text-only saliency already covers vision; prune as planned.
  SEPARATE -> text-only saliency is blind to vision; those experts need
              protecting (or vision data folded into the saliency mix).
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _mean_reap(entry):
    s = np.asarray(entry["reap_sum"], dtype=float)
    c = np.asarray(entry["reap_count"], dtype=float)
    return np.where(c > 0, s / np.maximum(c, 1), 0.0)


def _load(path):
    d = json.load(open(path))
    out = {}
    for k, v in d.items():
        if "reap_sum" in v:
            out[int(k)] = _mean_reap(v)
        elif "reap" in v:
            out[int(k)] = np.asarray(v["reap"], dtype=float)
    return out


def main(vision_path, text_path, plan_path, top_frac):
    vis = _load(vision_path)
    txt = _load(text_path)
    layers = sorted(set(vis) & set(txt))
    plan = json.load(open(plan_path)) if plan_path else None

    corrs, overlaps, pruned_hits, pruned_mass = [], [], [], []
    print(f"{'layer':>5} {'spearman':>9} {'top25%_overlap':>15} {'vis_top_in_prune':>17}")
    for l in layers:
        v, t = vis[l], txt[l]
        rv, rt = np.argsort(np.argsort(v)), np.argsort(np.argsort(t))
        corr = np.corrcoef(rv, rt)[0, 1]
        k = max(1, int(len(v) * top_frac))
        tv, tt = set(np.argsort(v)[::-1][:k].tolist()), set(np.argsort(t)[::-1][:k].tolist())
        ov = len(tv & tt) / k
        corrs.append(corr); overlaps.append(ov)

        hit = ""
        if plan is not None and str(l) in plan["layers"]:
            prune = set(plan["layers"][str(l)]["prune"])
            n_hit = len(tv & prune)
            pruned_hits.append(n_hit)
            # fraction of TOTAL vision saliency mass sitting on pruned experts
            pruned_mass.append(v[list(prune)].sum() / max(v.sum(), 1e-9))
            hit = f"{n_hit}/{k}"
        print(f"{l:>5} {corr:>9.3f} {ov:>15.3f} {hit:>17}")

    mc, mo = float(np.mean(corrs)), float(np.mean(overlaps))
    print(f"\nMEAN spearman={mc:.3f}  MEAN top{int(top_frac*100)}%_overlap={mo:.3f}")
    if pruned_hits:
        print(f"MEAN vision-top experts inside prune set: {np.mean(pruned_hits):.1f} "
              f"(max {max(pruned_hits)})")
        print(f"MEAN vision saliency MASS on pruned experts: "
              f"{100*np.mean(pruned_mass):.2f}%  (max {100*max(pruned_mass):.2f}%)")

    print("\nVERDICT:")
    if mc > 0.75 and mo > 0.6:
        print("  MIXED -- vision and text largely share experts. Text-only saliency")
        print("  already covers vision routing; current pruning approach is sound.")
    elif mc < 0.4 or mo < 0.35:
        print("  SEPARATE -- vision uses substantially different experts. Text-only")
        print("  saliency is BLIND to them; they must be protected from pruning or")
        print("  vision data must be folded into the saliency mix.")
    else:
        print("  PARTIAL overlap -- some shared, some vision-specific experts.")
        print("  Check the 'vision saliency MASS on pruned experts' number above:")
        print("  if it is more than a few percent, the current plan is costing vision.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", default="artifacts/step37-vision-saliency/saliency.json")
    ap.add_argument("--text", default="artifacts/step37-bf16-layerwise-5k/saliency.json")
    ap.add_argument("--plan", default="artifacts/step37-bf16-layerwise-5k/plan_p15.json",
                    help="text-only plan whose prune set we test against vision")
    ap.add_argument("--top-frac", type=float, default=0.25)
    a = ap.parse_args()
    main(a.vision, a.text, a.plan, a.top_frac)
