"""Per-prompt MoE routing capture for DeepSeek-V4 via oMLX's loader.

Why not `collect_routing_deepseek_v4.py`: that one goes through
``mlx_vlm.load``, which cannot build this AWQ checkpoint (it lacks the AWQ
weight transform and the per-expert `biases` stacking). oMLX registers
``deepseek_v4`` into **mlx_lm** and patches ``mlx_lm.utils.load_model``
accordingly, so applying that patch first makes ``mlx_lm.load`` work.

Routing is captured on ``omlx.patches.deepseek_v4.switch_layers.SwitchGLU``
(oMLX ships its own, distinct from ``mlx_lm.models.switch_layers``).

Records routing PER PROMPT so per-request concentration can be separated
from the pooled statistic in telemetry.json -- pooling across a diverse
calibration set flattens exactly the structure an AFM3-style pinned expert
set would exploit.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

_ROUTES: dict[int, list[np.ndarray]] = {}
_MARKS: list[tuple[str, int]] = []          # (prompt_tag, cumulative rows) per layer0


def install_capture():
    from omlx.patches.deepseek_v4.switch_layers import SwitchGLU
    orig = SwitchGLU.__call__
    ids: dict[int, int] = {}

    def patched(self, x, indices, *args, **kwargs):
        try:
            key = id(self)
            lay = ids.setdefault(key, len(ids))
            arr = np.array(indices, copy=False)
            _ROUTES.setdefault(lay, []).append(
                arr.reshape(-1, arr.shape[-1]).astype(np.int16))
        except Exception as e:
            print(f"[routing] capture failed: {e}", file=sys.stderr)
        return orig(self, x, indices, *args, **kwargs)

    SwitchGLU.__call__ = patched
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-file", required=True)
    ap.add_argument("--max-samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 = prefill only (routing over prompt tokens)")
    a = ap.parse_args()

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    ok = apply_deepseek_v4_patch()
    print(f"[routing] oMLX deepseek_v4 patch applied: {ok}", flush=True)

    install_capture()

    import mlx.core as mx
    from mlx_lm import load
    from reap_stream.dataset import load_prompt_texts

    prompts = load_prompt_texts(a.dataset_file, limit=a.max_samples)
    print(f"[routing] loading {a.model} ...", flush=True)
    model, tok = load(a.model)
    print("[routing] loaded", flush=True)

    boundaries = []
    for i, p in enumerate(prompts):
        ids = mx.array([tok.encode(p)])
        model(ids)
        mx.eval(mx.zeros(1))
        n = _ROUTES[0][-1].shape[0] if 0 in _ROUTES else 0
        boundaries.append(n)
        print(f"[routing] prompt {i}: {n} token-slots", flush=True)

    payload, layers = {}, sorted(_ROUTES)
    for lay in layers:
        payload[f"L{lay}"] = np.concatenate(_ROUTES[lay], axis=0)
    payload["_layers"] = np.array(layers, dtype=np.int32)
    payload["_bounds"] = np.array(boundaries, dtype=np.int64)
    payload["_meta"] = np.frombuffer(json.dumps({
        "model": a.model, "n_prompts": len(prompts),
        "per_prompt_rows": boundaries, "n_prompt_tokens": 0}).encode(), np.uint8)
    np.savez_compressed(a.out, **payload)
    print(f"[routing] wrote {a.out}: {len(layers)} layers", flush=True)


if __name__ == "__main__":
    main()
