"""Layer-streaming REAP saliency collection for Laguna-S-2.1 (poolside).

Port of collect_deepseek_v4.py's windowed streaming design to a structurally
simpler MoE architecture. Verified against the installed mlx_vlm source
(mlx_vlm/models/laguna/language.py), not assumed from the HF config alone.

Differences from DeepSeek-V4 that simplify this port:
1. No Hyper-Connections -- hidden state stays plain (batch, seq, hidden)
   throughout, same as Gemma-4.
2. No hash-routed layers -- every MoE layer uses learned top-k routing
   (LagunaTopKRouter: sigmoid-or-softmax + e_score_correction_bias, same
   correction-bias convention as MiniMax/DeepSeek but no fixed hash table).
3. Two attention types (full_attention / sliding_attention) with masks built
   ONCE up front (LagunaModel.__call__) and selected per layer by
   `layer.attention_type`, mirrored here exactly.

Still true, same as DeepSeek-V4 (verified against language.py, not assumed):
- Routing lives INSIDE the MoE block's __call__ (`self.gate(x)` then
  `self.switch_mlp(x, inds)` then `+ self.shared_expert(x)`), not computed
  upstream like Gemma-4's `layer.experts`. So the probe here wraps the whole
  LagunaSparseMoeBlock and reimplements its exact forward math, rather than
  reusing Gemma-4's simpler post-routing wrap.
- switch_mlp output shape convention (..., top_k, hidden) before the
  weighted sum, and LayerSaliency.update()'s (tokens, k) input contract, are
  identical -- saliency.py is architecture-agnostic and reused unchanged.

Layer 0 is dense (`mlp_only_layers: [0]` in config.json) -- not a MoE layer,
skipped automatically here by checking for a `switch_mlp` attribute rather
than trusting a hardcoded layer index (robust to config changes).

`num_experts` (256) and `num_experts_per_tok` (10) are single global config
fields -- every MoE layer must end up the SAME kept-expert width after
pruning (enforced in apply_laguna.py), same constraint as DeepSeek-V4's
`n_routed_experts`.

Usage:
    .venv/bin/python -m reap_stream.collect_laguna \
        --model ~/.lmstudio/models/poolside/Laguna-S-2.1-NVFP4-mlx \
        --output artifacts/laguna-reap \
        --dataset-file calib/cerebras_reap_mix.jsonl \
        --ratio 0.375 --mode layerwise --layers-at-once 1
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable, Literal, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask

from .saliency import LayerSaliency, build_plan
from .dataset import load_prompt_texts

CollectMode = Literal["full", "layerwise"]


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _mem_mb() -> dict[str, float]:
    return {
        "active_mb": mx.get_active_memory() / (1024**2),
        "peak_mb": mx.get_peak_memory() / (1024**2),
        "cache_mb": mx.get_cache_memory() / (1024**2),
    }


def _default_prompts() -> list[str]:
    return [
        "Write a Python function that merges two sorted lists.",
        "Explain how mixture-of-experts routing works in one paragraph.",
        "def fibonacci(n):\n    ",
        "What are three failure modes of tool-using agents?",
        "Summarize the tradeoffs between expert pruning and expert merging.",
        "Select the correct SQL join for matching customers to orders.",
        "Draft a minimal REST API in FastAPI with one health endpoint.",
        "Prove that the sum of the first n odds equals n squared.",
    ]


def _tokenize_prompts(tokenizer, prompts: list[str], max_tokens: int) -> list[list[int]]:
    batches: list[list[int]] = []
    for p in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text_in = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text_in = p
        else:
            text_in = p
        tokens = tokenizer.encode(text_in)[:max_tokens]
        batches.append(tokens)
    return batches


def _is_moe_layer(layer) -> bool:
    return hasattr(layer.mlp, "switch_mlp")


class _MoEProbe(nn.Module):
    """Wraps a LagunaSparseMoeBlock, reimplementing its __call__ verbatim
    (verified against mlx_vlm/models/laguna/language.py) so behavior is
    unchanged except for saliency bookkeeping. Cannot reuse Gemma-4's
    simpler post-routing wrap because routing (self.gate) happens inside
    this module, not before it -- same reason as DeepSeek-V4's _MoEProbe."""

    def __init__(self, inner: nn.Module, layer_idx: int, stats: dict[int, LayerSaliency]):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self._stats = stats

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.inner.gate(x)
        y = self.inner.switch_mlp(x, inds)

        norms = mx.sqrt((y.astype(mx.float32) ** 2).sum(axis=-1) + 1e-12)
        scores_f32 = scores.astype(mx.float32)
        mx.eval(inds, scores_f32, norms)
        ids_np = np.array(inds, dtype=np.int64).reshape(-1, inds.shape[-1])
        gates_np = np.array(scores_f32, dtype=np.float64).reshape(-1, scores_f32.shape[-1])
        norms_np = np.array(norms, dtype=np.float64).reshape(-1, norms.shape[-1])
        self._stats[self.layer_idx].update(ids_np, gates_np, norms_np)

        y = mx.sum(y * scores[..., None], axis=-2)
        if self.inner.routed_scaling_factor != 1.0:
            y = y * self.inner.routed_scaling_factor
        return y + self.inner.shared_expert(x)


class _DropLayer(nn.Module):
    """Placeholder after a layer has been scored and freed."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("freed layer was invoked — layerwise collector bug")


def _free_layer(text, layer_idx: int) -> None:
    text.layers[layer_idx] = _DropLayer()


def _run_layer(layer, h, mask):
    h_out = layer(h, mask, None)
    mx.eval(h_out)
    return h_out


def collect_full(
    model_path: str,
    prompts: Optional[Iterable[str]] = None,
    max_tokens: int = 256,
    layers: Optional[list[int]] = None,
) -> tuple[dict[int, LayerSaliency], list[dict]]:
    model, processor = load(model_path, lazy=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    text = _text_model(model)
    n_experts = text.args.num_experts
    moe_layer_ids = [i for i, l in enumerate(text.layers) if _is_moe_layer(l)]
    layer_ids = layers if layers is not None else moe_layer_ids
    stats = {i: LayerSaliency(num_experts=n_experts) for i in layer_ids}
    originals = {}
    for i in stats:
        originals[i] = text.layers[i].mlp
        text.layers[i].mlp = _MoEProbe(originals[i], i, stats)

    trace = [{"event": "loaded_full", **_mem_mb()}]
    prompts = list(prompts) if prompts is not None else _default_prompts()
    try:
        for tokens in _tokenize_prompts(tokenizer, prompts, max_tokens):
            out = model.language_model(mx.array(tokens)[None])
            mx.eval(out.logits if hasattr(out, "logits") else out)
            mx.clear_cache()
    finally:
        for i, mlp in originals.items():
            text.layers[i].mlp = mlp
    trace.append({"event": "done_full", **_mem_mb()})
    return stats, trace


def collect_layerwise(
    model_path: str,
    prompts: Optional[Iterable[str]] = None,
    max_tokens: int = 256,
    layers: Optional[list[int]] = None,
    layers_at_once: int = 1,
) -> tuple[dict[int, LayerSaliency], list[dict]]:
    """Windowed layer-wise REAP collection for Laguna-S-2.1.

    Same windowing strategy as collect_deepseek_v4.py's collect_layerwise:
    ``layers_at_once`` decoder blocks stay resident, all calibration batches
    run through that window, then the whole window is freed before the next.
    """
    if layers_at_once < 1:
        raise ValueError("layers_at_once must be >= 1")

    mx.reset_peak_memory()
    model, processor = load(model_path, lazy=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    text = _text_model(model)
    n_layers = len(text.layers)
    n_experts = text.args.num_experts
    sliding_window = text.args.sliding_window
    moe_layer_ids = {i for i, l in enumerate(text.layers) if _is_moe_layer(l)}

    target_layers = set(layers if layers is not None else moe_layer_ids)
    stats = {i: LayerSaliency(num_experts=n_experts) for i in target_layers}
    trace: list[dict] = [
        {"event": "loaded_lazy", "layers_at_once": layers_at_once, **_mem_mb()}
    ]

    prompts = list(prompts) if prompts is not None else _default_prompts()
    token_batches = _tokenize_prompts(tokenizer, prompts, max_tokens)

    hidden: list[mx.array] = []
    full_masks: list[mx.array] = []
    sliding_masks: list[mx.array] = []
    for tokens in token_batches:
        ids = mx.array(tokens)[None]
        h = text.embed_tokens(ids)
        full_mask = create_attention_mask(h, None)
        sliding_mask = create_attention_mask(h, None, window_size=sliding_window)
        mx.eval(h, full_mask, sliding_mask)
        hidden.append(h)
        full_masks.append(full_mask)
        sliding_masks.append(sliding_mask)
    trace.append({"event": "embedded", "batches": len(hidden), **_mem_mb()})

    for window_start in range(0, n_layers, layers_at_once):
        window = list(range(window_start, min(window_start + layers_at_once, n_layers)))

        for layer_idx in window:
            if layer_idx in stats:
                layer = text.layers[layer_idx]
                layer.mlp = _MoEProbe(layer.mlp, layer_idx, stats)

        for layer_idx in window:
            layer = text.layers[layer_idx]
            is_sliding = layer.attention_type == "sliding_attention"
            masks = sliding_masks if is_sliding else full_masks
            hidden = [
                _run_layer(layer, h, mask) for h, mask in zip(hidden, masks)
            ]

            mem = _mem_mb()
            hits = int(stats[layer_idx].freq.sum()) if layer_idx in stats else 0
            trace.append(
                {
                    "event": "layer_done",
                    "layer": layer_idx,
                    "window": window,
                    "expert_hits": hits,
                    **mem,
                }
            )
            print(
                f"[layerwise x{layers_at_once}] layer {layer_idx:02d}/{n_layers - 1} "
                f"window={window[0]}-{window[-1]} "
                f"active_mb={mem['active_mb']:.0f} peak_mb={mem['peak_mb']:.0f} "
                f"moe_hits={hits}",
                flush=True,
            )

        for layer_idx in window:
            _free_layer(text, layer_idx)
        gc.collect()
        mx.clear_cache()
        trace.append({"event": "window_freed", "window": window, **_mem_mb()})

    text.embed_tokens = nn.Identity()
    gc.collect()
    mx.clear_cache()
    trace.append({"event": "done_layerwise", **_mem_mb()})
    return stats, trace


def collect(
    model_path: str,
    prompts: Optional[Iterable[str]] = None,
    max_tokens: int = 256,
    layers: Optional[list[int]] = None,
    mode: CollectMode = "layerwise",
    layers_at_once: int = 1,
) -> tuple[dict[int, LayerSaliency], list[dict]]:
    if mode == "layerwise":
        return collect_layerwise(
            model_path, prompts, max_tokens, layers, layers_at_once=layers_at_once
        )
    if mode == "full":
        return collect_full(model_path, prompts, max_tokens, layers)
    raise ValueError(f"unknown mode {mode!r}")


def collect_and_plan(
    model_path: str,
    output_dir: str | Path,
    ratio: float = 0.25,
    prompts: Optional[Iterable[str]] = None,
    max_tokens: int = 256,
    layers: Optional[list[int]] = None,
    min_experts: int = 1,
    mode: CollectMode = "layerwise",
    layers_at_once: int = 1,
    dataset_file: Optional[str | Path] = None,
    max_samples: Optional[int] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_file is not None:
        prompts = load_prompt_texts(dataset_file, limit=max_samples)

    stats, trace = collect(
        model_path=model_path,
        prompts=prompts,
        max_tokens=max_tokens,
        layers=layers,
        mode=mode,
        layers_at_once=layers_at_once,
    )
    telemetry = {str(k): v.to_dict() for k, v in stats.items()}
    (output_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=2))
    (output_dir / "memory-trace.json").write_text(json.dumps(trace, indent=2))

    plan = build_plan(stats, ratio=ratio, min_experts=min_experts)
    plan["model_path"] = str(model_path)
    plan["collect_mode"] = mode
    plan["layers_at_once"] = layers_at_once
    plan_path = output_dir / "pruning-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    return plan_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--min-experts", type=int, default=1)
    ap.add_argument("--mode", choices=["full", "layerwise"], default="layerwise")
    ap.add_argument("--layers-at-once", type=int, default=1)
    ap.add_argument("--dataset-file", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args()

    plan_path = collect_and_plan(
        model_path=a.model,
        output_dir=a.output,
        ratio=a.ratio,
        max_tokens=a.max_tokens,
        layers=a.layers,
        min_experts=a.min_experts,
        mode=a.mode,
        layers_at_once=a.layers_at_once,
        dataset_file=a.dataset_file,
        max_samples=a.max_samples,
    )
    print(f"[collect] wrote -> {plan_path}", flush=True)


if __name__ == "__main__":
    main()
