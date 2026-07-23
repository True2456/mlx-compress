"""Streaming REAP collector for Step-3.7 Flash (mlx_vlm step3p7)."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask

from .dataset import load_prompt_texts
from .saliency import LayerSaliency, build_plan

CollectMode = Literal["full", "layerwise"]


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _text_config(model):
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "args"):
        return lm.args
    text = _text_model(model)
    return getattr(text, "args", None)


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
        "What are three failure modes of tool-using agents?",
        "Prove that the sum of the first n odds equals n squared.",
    ]


# Release MLX's buffer cache every N per-prompt forwards (see loops below).
_CACHE_EVERY = 200


def _truncate(tokens: list[int], max_tokens: int, mode: str) -> list[int]:
    """Truncate a token list to max_tokens under the given policy.

    head     : first max_tokens (original behavior; loses the ASSISTANT answer
               for long agentic/tool prompts whose setup exceeds the budget).
    tail     : last max_tokens (always includes the answer, loses early setup).
    headtail : first half + last half (keeps both task setup and the answer,
               with a discontinuity at the seam).
    """
    if len(tokens) <= max_tokens:
        return list(tokens)
    if mode == "tail":
        return list(tokens[-max_tokens:])
    if mode == "headtail":
        h = max_tokens // 2
        t = max_tokens - h
        return list(tokens[:h]) + list(tokens[-t:])
    return list(tokens[:max_tokens])  # head (default)


def _tokenize_prompts(processor, prompts: list[str], max_tokens: int,
                      truncation: str = "head") -> list[list[int]]:
    tokenizer = getattr(processor, "tokenizer", processor)
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
        tokens = tokenizer.encode(text_in)
        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]
        batches.append(_truncate(list(tokens), max_tokens, truncation))
    return batches


def _moe_layer_ids(text) -> list[int]:
    return [i for i, layer in enumerate(text.layers) if getattr(layer, "is_moe_layer", False)]


def _num_experts(text) -> int:
    for layer in text.layers:
        if getattr(layer, "is_moe_layer", False):
            return int(layer.mlp.gate.gate.weight.shape[0])
    cfg = getattr(text, "args", None)
    if cfg is not None:
        return int(getattr(cfg, "moe_num_experts", 0))
    raise RuntimeError("no MoE layers found")


class _MoEProbe(nn.Module):
    """Wrap Step MoE to accumulate REAP saliency without changing math."""

    def __init__(self, inner: nn.Module, layer_idx: int, stats: dict[int, LayerSaliency]):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self._stats = stats

    def __call__(self, x):
        topk_indices, topk_weights = self.inner.gate(x)
        y = self.inner.switch_mlp(x, topk_indices)
        norms = mx.sqrt((y.astype(mx.float32) ** 2).sum(axis=-1) + 1e-12)
        mx.eval(topk_indices, topk_weights, norms)
        ids = np.array(topk_indices, dtype=np.int64).reshape(-1, topk_indices.shape[-1])
        gates = np.array(topk_weights, dtype=np.float64).reshape(-1, topk_weights.shape[-1])
        nrm = np.array(norms, dtype=np.float64).reshape(-1, norms.shape[-1])
        self._stats[self.layer_idx].update(ids, gates, nrm)
        routed = (y * topk_weights[..., None]).sum(axis=-2).astype(y.dtype)
        return routed + self.inner.share_expert(x)


class _DropLayer(nn.Module):
    def __call__(self, h, *args, **kwargs):
        raise RuntimeError("freed layer was invoked — layerwise collector bug")


def _free_layer(text, layer_idx: int) -> None:
    text.layers[layer_idx] = _DropLayer()


def _run_layer(layer, h, sliding_window: int | None):
    if layer.is_sliding:
        mask = create_attention_mask(h, None, window_size=sliding_window)
    else:
        mask = create_attention_mask(h, None)
    h_out = layer(h, mask=mask, cache=None)
    mx.eval(h_out)
    return h_out


def inspect_model(model_path: str) -> dict:
    """Load lazy and report MoE layout (no forward)."""
    model, processor = load(model_path, lazy=True)
    text = _text_model(model)
    cfg = _text_config(model)
    moe_ids = _moe_layer_ids(text)
    n_experts = _num_experts(text) if moe_ids else 0
    sample = {}
    if moe_ids:
        moe = text.layers[moe_ids[0]].mlp
        sample = {
            "router_weight": list(moe.gate.gate.weight.shape),
            "router_bias": list(moe.gate.router_bias.shape),
            "switch_gate_proj": list(moe.switch_mlp.gate_proj.weight.shape),
            "share_expert_gate": list(moe.share_expert.gate_proj.weight.shape),
            "top_k": int(moe.gate.top_k),
            "norm_topk_prob": bool(moe.gate.norm_topk_prob),
            "routed_scaling_factor": float(moe.gate.routed_scaling_factor),
        }
    info = {
        "arch": "step3p7",
        "n_layers": len(text.layers),
        "moe_layers": moe_ids,
        "n_moe_layers": len(moe_ids),
        "moe_num_experts": n_experts,
        "sliding_window": getattr(cfg, "sliding_window", None),
        "sample_moe": sample,
        "has_vision": hasattr(model, "vision_model") or hasattr(model, "vision_tower"),
        **_mem_mb(),
    }
    del model, processor
    gc.collect()
    mx.clear_cache()
    return info


def _checkpoint_paths(checkpoint_dir: Path) -> tuple[Path, Path, Path]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return (
        checkpoint_dir / "state.json",
        checkpoint_dir / "saliency_partial.json",
        checkpoint_dir / "trace_partial.json",
    )


def _save_checkpoint(
    checkpoint_dir: Path,
    stats: dict[int, LayerSaliency],
    last_completed_layer: int,
    meta: dict,
    trace: list[dict],
) -> None:
    state_path, sal_path, trace_path = _checkpoint_paths(checkpoint_dir)
    sal_path.write_text(
        json.dumps({str(k): v.to_checkpoint_dict() for k, v in stats.items()})
    )
    state_path.write_text(
        json.dumps({"last_completed_layer": last_completed_layer, **meta}, indent=2)
    )
    trace_path.write_text(json.dumps(trace, indent=2))


def _load_checkpoint(
    checkpoint_dir: Path,
    meta: dict,
) -> tuple[int, dict[int, LayerSaliency], list[dict]] | None:
    state_path, sal_path, trace_path = _checkpoint_paths(checkpoint_dir)
    if not state_path.exists() or not sal_path.exists():
        return None
    state = json.loads(state_path.read_text())
    for key in ("model_path", "max_tokens", "layers_at_once", "n_prompts"):
        if state.get(key) != meta.get(key):
            return None
    stats = {
        int(k): LayerSaliency.from_checkpoint_dict(v)
        for k, v in json.loads(sal_path.read_text()).items()
    }
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
    return int(state["last_completed_layer"]), stats, trace


def collect_layerwise(
    model_path: str,
    prompts: Optional[Iterable[str]] = None,
    max_tokens: int = 128,
    layers: Optional[list[int]] = None,
    layers_at_once: int = 1,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
    truncation: str = "head",
) -> tuple[dict[int, LayerSaliency], list[dict]]:
    """Carry hidden states across windows; free each window after scoring."""
    if layers_at_once < 1:
        raise ValueError("layers_at_once must be >= 1")

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    prompt_list = list(prompts) if prompts is not None else _default_prompts()
    run_meta = {
        "model_path": str(model_path),
        "max_tokens": max_tokens,
        "layers_at_once": layers_at_once,
        "n_prompts": len(prompt_list),
    }

    resume_after = -1
    restored_stats: dict[int, LayerSaliency] | None = None
    trace: list[dict] = []
    if ckpt_dir and resume:
        loaded = _load_checkpoint(ckpt_dir, run_meta)
        if loaded is not None:
            resume_after, restored_stats, trace = loaded
            trace.append({"event": "resume", "resume_after": resume_after, **_mem_mb()})
            print(
                f"[step3p7] resuming after layer {resume_after} "
                f"({len(restored_stats)} MoE layers checkpointed)",
                flush=True,
            )

    mx.reset_peak_memory()
    model, processor = load(model_path, lazy=True)
    text = _text_model(model)
    cfg = _text_config(model)
    sliding_window = getattr(cfg, "sliding_window", None)
    n_layers = len(text.layers)
    n_experts = _num_experts(text)
    moe_ids = set(_moe_layer_ids(text))

    if layers is None:
        # Smoke / full: all MoE layers. Dense 0–2 still run for hidden carry.
        target_moe = set(moe_ids)
        run_through = n_layers
    else:
        target_moe = set(layers)
        bad = target_moe - moe_ids
        if bad:
            raise ValueError(f"non-MoE layers requested: {sorted(bad)}")
        run_through = max(target_moe) + 1

    stats = {i: LayerSaliency(num_experts=n_experts) for i in sorted(target_moe)}
    if restored_stats:
        for layer_idx, layer_stats in restored_stats.items():
            stats[layer_idx] = layer_stats
    if not trace:
        trace = [
            {
                "event": "loaded_lazy",
                "layers_at_once": layers_at_once,
                "n_experts": n_experts,
                "target_moe": sorted(target_moe),
                "run_through": run_through,
                **_mem_mb(),
            }
        ]

    token_batches = _tokenize_prompts(processor, prompt_list, max_tokens, truncation)

    hidden: list[mx.array] = []
    for tokens in token_batches:
        h = text.embed_tokens(mx.array(tokens)[None])
        mx.eval(h)
        hidden.append(h)
    trace.append({"event": "embedded", "batches": len(hidden), **_mem_mb()})

    if resume_after >= 0:
        for layer_idx in range(0, resume_after + 1):
            layer = text.layers[layer_idx]
            # Update in place: a list comprehension would materialise an entire
            # second hidden-state list (~21GB at 2500x1024) before releasing the
            # old one, doubling peak memory. Replacing element-by-element frees
            # each old array as it goes.
            for i in range(len(hidden)):
                hidden[i] = _run_layer(layer, hidden[i], sliding_window)
                # MLX caches freed buffers instead of returning them to the OS.
                # Over 2500 per-prompt forwards that cache grows unbounded and
                # drives the machine into compression/swap. Cap it periodically.
                if (i + 1) % _CACHE_EVERY == 0:
                    mx.clear_cache()
            _free_layer(text, layer_idx)
            gc.collect()
            mx.clear_cache()
            trace.append(
                {
                    "event": "replay_layer",
                    "layer": layer_idx,
                    **_mem_mb(),
                }
            )
        gc.collect()
        mx.clear_cache()

    start_layer = resume_after + 1
    for window_start in range(start_layer, run_through, layers_at_once):
        window = list(range(window_start, min(window_start + layers_at_once, run_through)))

        for layer_idx in window:
            if layer_idx in stats and layer_idx > resume_after:
                layer = text.layers[layer_idx]
                layer.mlp = _MoEProbe(layer.mlp, layer_idx, stats)

        for layer_idx in window:
            layer = text.layers[layer_idx]
            # in-place update + periodic cache release -- see replay loop above
            for i in range(len(hidden)):
                hidden[i] = _run_layer(layer, hidden[i], sliding_window)
                if (i + 1) % _CACHE_EVERY == 0:
                    mx.clear_cache()
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
                f"[step3p7 x{layers_at_once}] layer {layer_idx:02d}/{run_through - 1} "
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
        if ckpt_dir is not None:
            _save_checkpoint(ckpt_dir, stats, window[-1], run_meta, trace)
            print(
                f"[step3p7] checkpoint saved after layer {window[-1]}",
                flush=True,
            )

    text.embed_tokens = nn.Identity()
    gc.collect()
    mx.clear_cache()
    trace.append({"event": "done_layerwise", **_mem_mb()})
    del model, processor
    gc.collect()
    mx.clear_cache()
    return stats, trace


def collect_and_plan(
    model_path: str,
    output_dir: str | Path,
    ratio: float = 0.10,
    max_tokens: int = 128,
    layers: Optional[list[int]] = None,
    min_experts: int = 8,
    mode: CollectMode = "layerwise",
    layers_at_once: int = 1,
    dataset_file: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Path:
    if mode != "layerwise":
        raise NotImplementedError("step3p7 collector supports layerwise only for now")

    prompts = None
    if dataset_file:
        prompts = load_prompt_texts(dataset_file, limit=max_samples)

    stats, trace = collect_layerwise(
        model_path,
        prompts=prompts,
        max_tokens=max_tokens,
        layers=layers,
        layers_at_once=layers_at_once,
    )
    plan = build_plan(stats, ratio=ratio, min_experts=min_experts)
    plan["arch"] = "step3p7"
    plan["model_path"] = str(model_path)
    plan["mode"] = mode
    plan["layers_at_once"] = layers_at_once
    plan["smoke"] = True
    plan["note"] = (
        "Infrastructure smoke only — do not ship a prune of an already-REAPed checkpoint"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "pruning-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    (out / "saliency.json").write_text(
        json.dumps({str(k): v.to_dict() for k, v in stats.items()}, indent=2)
    )
    (out / "trace.json").write_text(json.dumps(trace, indent=2))
    return plan_path
