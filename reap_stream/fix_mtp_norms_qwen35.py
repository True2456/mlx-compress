"""Repair mixed norm conventions in a Qwen3.5-family MTP head.

Qwen3-Next stores every RMSNorm gamma zero-centred; MLX wants them centred on
one, so conversion adds 1.0. Our AWQ builds carry the MTP head across from the
*raw HF* source (``carry_mtp_weights.py``) after the backbone has already been
converted — so the checkpoint ends up mixed:

    language_model.model.layers.0.input_layernorm   0.9666   <- MLX convention
    mtp.layers.0.input_layernorm                    0.0361   <- raw HF

Nothing downstream can fix this reliably. ``mlx_vlm``'s ``sanitize`` is skipped
entirely for checkpoints already tagged ``format: mlx``; the MTP drafter split
copies tensors through untouched; and oMLX's ``norm_repair`` falls back to a
``mean < 0.5`` heuristic that catches the four low norms and misses ``q_norm``
(0.79), ``k_norm`` (0.79) and ``mtp.norm`` (1.25) — a misclassification oMLX's
own source notes costs about 14 points of draft acceptance.

The consequence is not a crash. The head's first layernorm multiplies by 0.036
instead of ~1.036, drafts stop matching the target, acceptance collapses, and
MTP becomes pure overhead — measured as decode *slower* with MTP on (22.4 tok/s)
than off (24.6).

This writes a **new** checkpoint; the source is never modified. Other shards are
cloned with APFS copy-on-write, so the copy costs no extra disk until touched.

    python -m reap_stream.fix_mtp_norms_qwen35 --model <src> --output <dst>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

# The convention is a property of the whole head, not of individual tensors, so
# the decision is made once from the strongest evidence and then applied
# uniformly. Judging each norm on its own mean is precisely the bug this script
# exists to fix: raw `q_norm`/`k_norm` average ~0.78 and raw `mtp.norm` ~1.25,
# so any per-key threshold that catches `input_layernorm` (0.04) leaves those
# three behind — which is how oMLX's norm_repair loses ~14pp of acceptance.
#
# A raw-HF head is proven by its *lowest* norm: zero-centred gammas produce
# means at or below zero, which a one-centred gamma cannot.
RAW_HF_MIN_MEAN = 0.5


def _mtp_norm_keys(weights: dict) -> list[str]:
    return sorted(
        k
        for k, v in weights.items()
        if k.startswith("mtp.") and k.endswith(".weight") and v.ndim == 1
    )


def _clone_tree(src: Path, dst: Path) -> None:
    """APFS copy-on-write clone, falling back to a real copy off APFS."""
    if dst.exists():
        raise SystemExit(f"output already exists: {dst}")
    try:
        subprocess.run(["cp", "-c", "-R", str(src), str(dst)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.copytree(src, dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fix-mtp-norms", description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the norms and what would change, write nothing.",
    )
    args = ap.parse_args(argv)

    src = Path(args.model)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({index[k] for k in index if k.startswith("mtp.")})
    if not shards:
        print(f"[fix-mtp-norms] no mtp.* tensors in {src}", file=sys.stderr)
        return 2

    # Inspect the whole head first, so the convention is decided once and a
    # already-converted checkpoint is rejected before anything is written.
    plan: dict[str, list[str]] = {}
    means: dict[str, float] = {}
    for shard in shards:
        weights = mx.load(str(src / shard))
        keys = _mtp_norm_keys(weights)
        for key in keys:
            means[key] = float(mx.mean(weights[key].astype(mx.float32)).item())
        if keys:
            plan[shard] = keys

    if not means:
        print(f"[fix-mtp-norms] no 1-D mtp norms in {src}", file=sys.stderr)
        return 2

    lowest = min(means.values())
    for key in sorted(means):
        print(f"[fix-mtp-norms] {key:50s} mean={means[key]:+.4f} -> {means[key] + 1:+.4f}")

    if lowest > RAW_HF_MIN_MEAN:
        print(
            f"[fix-mtp-norms] REFUSING: lowest head norm averages {lowest:.4f}. "
            "Every gamma is already one-centred, so this head looks converted; "
            "shifting again would corrupt it as silently as leaving it raw.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[fix-mtp-norms] head is raw-HF (lowest norm {lowest:+.4f}); "
        f"shifting all {len(means)} norms uniformly by +1.0"
    )

    total = sum(len(v) for v in plan.values())
    if args.dry_run:
        print(f"[fix-mtp-norms] dry run: would shift {total} norms in {len(plan)} shard(s)")
        return 0

    dst = Path(args.output)
    _clone_tree(src, dst)

    for shard, keys in plan.items():
        weights = mx.load(str(dst / shard))
        for key in keys:
            weights[key] = weights[key] + 1.0
        # mx.load returns arrays still backed by the source file. Writing to that
        # same path truncates it before the lazy reads happen and every untouched
        # tensor comes back as zeros. Materialise, then write via a temp file.
        mx.eval(list(weights.values()))
        tmp = dst / (shard + ".tmp")
        mx.save_safetensors(str(tmp), weights, metadata={"format": "mlx"})
        # MLX appends .safetensors to whatever path it is given.
        written = tmp.with_name(tmp.name + ".safetensors")
        written.replace(dst / shard)
        print(f"[fix-mtp-norms] rewrote {shard} ({len(keys)} norms)")

    # Verify from disk rather than trusting the in-memory arrays — the lazy-write
    # trap above fails by zeroing untouched tensors, which only a re-read catches.
    print("[fix-mtp-norms] verifying:")
    ok = True
    for shard, keys in plan.items():
        weights = mx.load(str(dst / shard))
        for key in keys:
            mean = float(mx.mean(weights[key].astype(mx.float32)).item())
            expected = means[key] + 1.0
            good = abs(mean - expected) < 1e-3
            ok = ok and good
            print(
                f"    {key:50s} mean={mean:+.4f} expected={expected:+.4f}"
                f"  {'ok' if good else 'MISMATCH'}"
            )
    # A zeroed neighbour is the classic symptom of the lazy-write trap, so spot
    # check a tensor in the same shard that we did not intend to touch.
    for shard in plan:
        weights = mx.load(str(dst / shard))
        others = [k for k, v in weights.items() if k not in means and v.ndim == 1]
        if others:
            probe = others[0]
            m = float(mx.mean(mx.abs(weights[probe].astype(mx.float32))).item())
            print(f"    untouched probe {probe}: mean|w|={m:.4f}"
                  f"  {'ok' if m > 0 else 'ZEROED — REWRITE CORRUPTED THE SHARD'}")
            ok = ok and m > 0
    print(f"[fix-mtp-norms] wrote {dst}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
