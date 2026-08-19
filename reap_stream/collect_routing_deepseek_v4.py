"""Capture MoE routing decisions during the streaming REAP sweep.

``collect_deepseek_v4._MoEProbe`` already computes ``inds`` and materialises
them as numpy on its way to the saliency accumulator, then throws them away.
This wraps that probe to also keep them, so a routing log costs one extra
array copy on top of a collection run that already streams the model one
decoder block at a time (``layers_at_once=1``).

That matters: observing routing needs a real forward pass, but it does NOT
need the model resident. A full load of DeepSeek-V4-Flash-AWQ is 108 GB
against ~108 GB free; this runs in the same footprint as the REAP collector.

Output is an ``.npz`` consumable by
``q38_native_engine/scratch/moe_route/moe_route_analyze.py``.

Usage:
    python -m reap_stream.collect_routing_deepseek_v4 \
        --model /path/to/DeepSeek-V4-Flash-0731-AWQ \
        --out routes.npz --dataset-file calib.jsonl --max-samples 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reap_stream import collect_deepseek_v4 as C


_ROUTES: dict[int, list[np.ndarray]] = {}
_PROMPT_TOKENS = {"n": 0}


def _install_biases_fix() -> None:
    """Stack per-expert `.biases` that mlx_vlm's sanitize() drops.

    language.py stacks experts for ("weight", "scales") but not "biases", so
    an affine-quantised AWQ build fails load_weights with 33024 unexpected
    params (43 layers x 256 experts x 3 projections). The adjacent attn.wo_a
    loop does include biases -- this is the known omission, not a design
    choice. Patched here rather than in the app bundle, which oMLX updates
    overwrite.
    """
    import mlx.core as mx
    from mlx_vlm.models.deepseek_v4 import language as LG

    for cls_name in ("LanguageModel", "Model"):
        cls = getattr(LG, cls_name, None)
        if cls is None or not hasattr(cls, "sanitize"):
            continue
        orig = cls.sanitize

        def patched(self, weights, _orig=orig):
            w = _orig(self, weights)
            n_layers = self.args.num_hidden_layers if hasattr(self, "args") else 0
            n_exp = getattr(self.args, "n_routed_experts", 0) if hasattr(self, "args") else 0
            for layer_idx in range(n_layers):
                prefix = f"model.layers.{layer_idx}.ffn.experts"
                for src, dst in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
                    key0 = f"{prefix}.0.{src}.biases"
                    if key0 not in w:
                        continue
                    stacked = [w.pop(f"{prefix}.{e}.{src}.biases") for e in range(n_exp)]
                    w[f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.biases"] = mx.stack(stacked)
            return w

        cls.sanitize = patched
        print(f"[routing] biases fix installed on {cls_name}", flush=True)


def _install_routing_capture() -> None:
    """Tee routing ids into _ROUTES without recomputing the gate.

    ``_MoEProbe.__call__`` hands ``ids_np`` straight to
    ``LayerSaliency.update``. Stashing it there costs one attribute write and
    guarantees we record exactly the array the saliency accumulator saw --
    no duplicated forward math that could drift from language.py.
    """
    orig_update = C.LayerSaliency.update
    orig_call = C._MoEProbe.__call__

    def patched_update(self, expert_ids, gate_weights, activation_norms):
        self._last_ids = expert_ids
        return orig_update(self, expert_ids, gate_weights, activation_norms)

    def patched_call(self, x, input_ids):
        out = orig_call(self, x, input_ids)
        try:
            stat = self._stats[self.layer_idx]
            ids = getattr(stat, "_last_ids", None)
            if ids is not None:
                _ROUTES.setdefault(self.layer_idx, []).append(
                    np.asarray(ids, dtype=np.int16)
                )
                stat._last_ids = None
        except Exception as e:  # never break a collection run
            print(f"[routing] layer {self.layer_idx} capture failed: {e}", flush=True)
        return out

    C.LayerSaliency.update = patched_update
    C._MoEProbe.__call__ = patched_call


def save(out_path: str | Path, meta: dict) -> None:
    out_path = Path(out_path)
    if not _ROUTES:
        raise SystemExit("no routing captured — did the probe install?")
    payload, layers = {}, sorted(_ROUTES)
    for lay in layers:
        payload[f"L{lay}"] = np.concatenate(_ROUTES[lay], axis=0)
    payload["_layers"] = np.array(layers, dtype=np.int32)
    payload["_meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(out_path, **payload)
    first = payload[f"L{layers[0]}"]
    print(f"[routing] wrote {out_path}: {len(layers)} layers, "
          f"{first.shape[0]} token-slots/layer, top{first.shape[1]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--layers-at-once", type=int, default=1)
    ap.add_argument("--dataset-file", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args()

    _install_biases_fix()
    _install_routing_capture()

    prompts = None
    if a.dataset_file:
        prompts = C.load_prompt_texts(a.dataset_file, limit=a.max_samples)
        prompts = list(prompts)
        _PROMPT_TOKENS["n"] = len(prompts)

    C.collect(
        model_path=a.model,
        prompts=prompts,
        max_tokens=a.max_tokens,
        layers=a.layers,
        mode="layerwise",
        layers_at_once=a.layers_at_once,
    )

    save(a.out, {
        "model": a.model,
        "n_prompt_tokens": 0,          # layerwise sweep: no prefill/decode split
        "max_tokens": a.max_tokens,
        "n_prompts": _PROMPT_TOKENS["n"],
        "note": "layerwise streaming sweep; split defaults to T//2 in the analyser",
    })


if __name__ == "__main__":
    main()
