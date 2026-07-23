from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable, Literal, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

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


class _ExpertsProbe(nn.Module):
    def __init__(self, inner: nn.Module, layer_idx: int, stats: dict[int, LayerSaliency]):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self._stats = stats

    def __call__(self, x, top_k_indices, top_k_weights):
        y = self.inner.switch_glu(x, top_k_indices)
        norms = mx.sqrt((y.astype(mx.float32) ** 2).sum(axis=-1) + 1e-12)
        mx.eval(top_k_indices, top_k_weights, norms)
        ids = np.array(top_k_indices, dtype=np.int64).reshape(-1, top_k_indices.shape[-1])
        gates = np.array(top_k_weights, dtype=np.float64).reshape(-1, top_k_weights.shape[-1])
        nrm = np.array(norms, dtype=np.float64).reshape(-1, norms.shape[-1])
        self._stats[self.layer_idx].update(ids, gates, nrm)
        w = mx.expand_dims(top_k_weights, -1)
        return (w * y).sum(-2)


class _DropLayer(nn.Module):
    """Placeholder after a layer has been scored and freed."""

    def __call__(self, h, *args, **kwargs):
        raise RuntimeError("freed layer was invoked — layerwise collector bug")


def _free_layer(text, layer_idx: int) -> None:
    text.layers[layer_idx] = _DropLayer()


def _run_layer(layer, h, window_size):
    if layer.layer_type == "sliding_attention":
        mask = create_attention_mask(h, None, window_size=window_size)
    else:
        mask = create_attention_mask(h, None)
    h_out, _, _ = layer(h, mask=mask, cache=None)
    mx.eval(h_out)
    return h_out


def collect_full(
    model_path: str,
    prompts: Optional[Iterable[str]] = None,
    max_tokens: int = 256,
    layers: Optional[list[int]] = None,
) -> tuple[dict[int, LayerSaliency], list[dict]]:
    model, tokenizer = load(model_path, lazy=False)
    text = _text_model(model)
    n_experts = text.layers[0].config.num_experts
    layer_ids = layers if layers is not None else list(range(len(text.layers)))
    stats = {
        i: LayerSaliency(num_experts=n_experts)
        for i in layer_ids
        if text.layers[i].enable_moe
    }
    originals = {}
    for i in stats:
        originals[i] = text.layers[i].experts
        text.layers[i].experts = _ExpertsProbe(originals[i], i, stats)

    trace = [{"event": "loaded_full", **_mem_mb()}]
    prompts = list(prompts) if prompts is not None else _default_prompts()
    try:
        for tokens in _tokenize_prompts(tokenizer, prompts, max_tokens):
            out = model(mx.array(tokens)[None])
            mx.eval(out)
            mx.clear_cache()
    finally:
        for i, experts in originals.items():
            text.layers[i].experts = experts
    trace.append({"event": "done_full", **_mem_mb()})
    return stats, trace


def collect_layerwise(
    model_path: str,
    prompts: Optional[Iterable[str]] = None,
    max_tokens: int = 256,
    layers: Optional[list[int]] = None,
    layers_at_once: int = 1,
) -> tuple[dict[int, LayerSaliency], list[dict]]:
    """Windowed layer-wise REAP collection.

    Keeps ``layers_at_once`` decoder blocks resident, runs all calib batches
    through that window, then frees the whole window before the next.
    ``layers_at_once=1`` is minimum memory; 2–4 is usually the speed sweet spot
    on 128 GB unified memory for large MoEs.
    """
    if layers_at_once < 1:
        raise ValueError("layers_at_once must be >= 1")

    mx.reset_peak_memory()
    model, tokenizer = load(model_path, lazy=True)
    text = _text_model(model)
    n_layers = len(text.layers)
    n_experts = text.layers[0].config.num_experts
    if getattr(text.config, "num_kv_shared_layers", 0):
        raise NotImplementedError(
            "layerwise collector does not yet support kv-shared layers"
        )
    if getattr(text, "hidden_size_per_layer_input", 0):
        raise NotImplementedError(
            "layerwise collector does not yet support per-layer input embeddings"
        )

    target_layers = set(layers if layers is not None else range(n_layers))
    stats = {
        i: LayerSaliency(num_experts=n_experts)
        for i in target_layers
        if text.layers[i].enable_moe
    }
    trace: list[dict] = [
        {"event": "loaded_lazy", "layers_at_once": layers_at_once, **_mem_mb()}
    ]

    prompts = list(prompts) if prompts is not None else _default_prompts()
    token_batches = _tokenize_prompts(tokenizer, prompts, max_tokens)

    hidden: list[mx.array] = []
    for tokens in token_batches:
        ids = mx.array(tokens)[None]
        h = text.embed_tokens(ids) * text.embed_scale
        mx.eval(h)
        hidden.append(h)
    trace.append({"event": "embedded", "batches": len(hidden), **_mem_mb()})

    window_size = getattr(text, "window_size", None)

    for window_start in range(0, n_layers, layers_at_once):
        window = list(range(window_start, min(window_start + layers_at_once, n_layers)))

        # Install probes for MoE layers in this window.
        for layer_idx in window:
            if layer_idx in stats:
                layer = text.layers[layer_idx]
                layer.experts = _ExpertsProbe(layer.experts, layer_idx, stats)

        # Run all batches through the whole window while layers stay resident.
        for layer_idx in window:
            layer = text.layers[layer_idx]
            hidden = [_run_layer(layer, h, window_size) for h in hidden]

            mem = _mem_mb()
            hits = int(stats[layer_idx].freq.sum()) if layer_idx in stats else 0
            trace.append(
                {
                    "event": "layer_done",
                    "layer": layer_idx,
                    "window": window,
                    "moe": layer_idx in stats,
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

        # Free the entire window before opening the next.
        for layer_idx in window:
            _free_layer(text, layer_idx)
        gc.collect()
        mx.clear_cache()
        trace.append(
            {
                "event": "window_freed",
                "window": window,
                **_mem_mb(),
            }
        )

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
