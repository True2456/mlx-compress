"""Layer-streaming REAP saliency collection for DeepSeek-V4-Flash.

Port of collect_gemma4.py's windowed streaming design to a structurally
different MoE architecture. Verified against the installed mlx_vlm source
(mlx_vlm/models/deepseek_v4/language.py) rather than assumed from the HF
config alone -- three real differences from Gemma-4 that change the port,
plus a few near-identical assumptions Gemma-4's collector relied on:

1. Loading: mlx_lm has no deepseek_v4 entry (only up to deepseek_v32) --
   mlx_vlm does, and its handling of this checkpoint's native HF fp8 format
   is real (mlx_vlm/utils.py, `quant_method == "fp8" and model_type ==
   "deepseek_v4"` routes to make_quantization_config), not a stub. So this
   loads via mlx_vlm, unlike collect_gemma4.py which uses plain mlx_lm.

2. Hyper-Connections: every layer's hidden state is NOT (batch, seq, hidden)
   like Gemma-4 -- it's (batch, seq, hc_mult, hidden), a set of hc_mult=4
   parallel residual streams only collapsed back to one at the very end via
   DeepseekV4Model.hc_head. This has to be built correctly once at the
   embedding stage (mx.broadcast_to + mx.contiguous, mirroring
   DeepseekV4Model.__call__ exactly) or every downstream layer computes
   garbage. The good news: HyperConnection state (attn_hc/ffn_hc) is fully
   local to each DeepseekV4Block.__call__ -- nothing threads between layers
   beyond h itself, so the streaming windowed-freeing design still applies
   unchanged in shape.

3. MoE routing lives INSIDE the FFN module, not before it. Gemma-4's
   collector wraps `layer.experts` (already receives top_k_indices/weights
   computed upstream). DeepSeek-V4's DeepseekV4MoE.__call__ computes routing
   itself (`self.gate(x, input_ids)`) then calls `self.switch_mlp(x, inds)`
   before a weighted sum + shared_experts additive term. So the probe here
   wraps the whole DeepseekV4MoE module and reimplements its exact forward
   math (verified against language.py's DeepseekV4MoE.__call__ line for
   line) rather than reusing Gemma-4's simpler post-routing wrap.

Near-identical to Gemma-4, verified not assumed:
- Mask: unlike Gemma-4 (two layer types, mask rebuilt per layer), DeepSeek-V4
  builds ONE mask up front in DeepseekV4Model.__call__ and reuses it for
  every layer -- simpler here, not harder.
- switch_mlp output shape convention (..., top_k, hidden) before the
  weighted sum, and LayerSaliency.update()'s (tokens, k) input contract, are
  identical to Gemma-4's -- saliency.py is architecture-agnostic and is
  reused unchanged.
- No embed_scale multiply: DeepseekV4Model.__call__ does NOT scale embeddings
  by sqrt(hidden_size) the way Gemma-4 does. Copying that detail over would
  be a silent correctness bug.

NOT yet handled, flagged rather than silently ignored:
- The first `num_hash_layers` (3) layers route via a fixed token->expert hash
  table (MoEGate.hash), not learned routing. Saliency still accumulates
  meaningfully there (real experts, real gate weights, real output norms),
  but a REAP prune plan for those layers reflects a fixed hash assignment,
  not a fine-tunable preference -- worth flagging to whoever consumes the
  plan, not a collection-time blocker.
- Pipeline/distributed sharding (sharding_group, pipeline_rank) is ignored --
  this collector assumes a single-process run, matching Gemma-4's collector.
- No apply_deepseek_v4.py yet (the pruning-plan -> pruned-checkpoint step,
  paralleling apply_gemma4.py). That needs its own design pass, particularly
  for the hash-routed layers' tid2eid table, which has no Gemma-4 analogue.

Usage:
    .venv/bin/python -m reap_stream.collect_deepseek_v4 \
        --model models/DeepSeek-V4-Flash-fp8 \
        --output artifacts/deepseek-v4-reap \
        --mode layerwise --layers-at-once 1
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


def _expand_hc(h: mx.array, hc_mult: int) -> mx.array:
    """(batch, seq, hidden) -> (batch, seq, hc_mult, hidden), matching
    DeepseekV4Model.__call__ exactly. Every layer downstream expects this
    shape; skipping it is a silent, not a loud, bug."""
    h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], hc_mult, h.shape[2]))
    return mx.contiguous(h)


class _MoEProbe(nn.Module):
    """Wraps a DeepseekV4MoE module, reimplementing its __call__ verbatim
    (verified against language.py) so behavior is unchanged except for the
    saliency bookkeeping. Cannot reuse Gemma-4's _ExpertsProbe as-is because
    routing (self.gate) happens inside this module, not before it."""

    def __init__(self, inner: nn.Module, layer_idx: int, stats: dict[int, LayerSaliency]):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self._stats = stats

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        inds, scores = self.inner.gate(x, input_ids)
        y = self.inner.switch_mlp(x, inds)

        norms = mx.sqrt((y.astype(mx.float32) ** 2).sum(axis=-1) + 1e-12)
        mx.eval(inds, scores, norms)
        ids_np = np.array(inds, dtype=np.int64).reshape(-1, inds.shape[-1])
        gates_np = np.array(scores, dtype=np.float64).reshape(-1, scores.shape[-1])
        norms_np = np.array(norms, dtype=np.float64).reshape(-1, norms.shape[-1])
        self._stats[self.layer_idx].update(ids_np, gates_np, norms_np)

        y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        y = y + self.inner.shared_experts(x)
        return y


class _DropLayer(nn.Module):
    """Placeholder after a layer has been scored and freed."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("freed layer was invoked — layerwise collector bug")


def _free_layer(text, layer_idx: int) -> None:
    text.layers[layer_idx] = _DropLayer()


def _run_layer(layer, h, mask, input_ids):
    h_out = layer(h, mask, None, input_ids)
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
    n_experts = text.args.n_routed_experts
    layer_ids = layers if layers is not None else list(range(len(text.layers)))
    stats = {i: LayerSaliency(num_experts=n_experts) for i in layer_ids}
    originals = {}
    for i in stats:
        originals[i] = text.layers[i].ffn
        text.layers[i].ffn = _MoEProbe(originals[i], i, stats)

    trace = [{"event": "loaded_full", **_mem_mb()}]
    prompts = list(prompts) if prompts is not None else _default_prompts()
    try:
        for tokens in _tokenize_prompts(tokenizer, prompts, max_tokens):
            out = model.language_model(mx.array(tokens)[None])
            mx.eval(out.logits if hasattr(out, "logits") else out)
            mx.clear_cache()
    finally:
        for i, ffn in originals.items():
            text.layers[i].ffn = ffn
    trace.append({"event": "done_full", **_mem_mb()})
    return stats, trace


def collect_layerwise(
    model_path: str,
    prompts: Optional[Iterable[str]] = None,
    max_tokens: int = 256,
    layers: Optional[list[int]] = None,
    layers_at_once: int = 1,
) -> tuple[dict[int, LayerSaliency], list[dict]]:
    """Windowed layer-wise REAP collection for DeepSeek-V4-Flash.

    Same windowing strategy as collect_gemma4.py's collect_layerwise:
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
    n_experts = text.args.n_routed_experts
    hc_mult = text.args.hc_mult
    sliding_window = text.args.sliding_window

    target_layers = set(layers if layers is not None else range(n_layers))
    stats = {i: LayerSaliency(num_experts=n_experts) for i in target_layers}
    trace: list[dict] = [
        {"event": "loaded_lazy", "layers_at_once": layers_at_once, **_mem_mb()}
    ]

    prompts = list(prompts) if prompts is not None else _default_prompts()
    token_batches = _tokenize_prompts(tokenizer, prompts, max_tokens)

    hidden: list[mx.array] = []
    input_ids_list: list[mx.array] = []
    masks: list[mx.array] = []
    for tokens in token_batches:
        ids = mx.array(tokens)[None]
        h = _expand_hc(text.embed_tokens(ids), hc_mult)
        mask = create_attention_mask(
            h[:, :, 0, :], None, window_size=sliding_window, return_array=True
        )
        mx.eval(h, mask)
        hidden.append(h)
        input_ids_list.append(ids)
        masks.append(mask)
    trace.append({"event": "embedded", "batches": len(hidden), **_mem_mb()})

    for window_start in range(0, n_layers, layers_at_once):
        window = list(range(window_start, min(window_start + layers_at_once, n_layers)))

        for layer_idx in window:
            if layer_idx in stats:
                layer = text.layers[layer_idx]
                layer.ffn = _MoEProbe(layer.ffn, layer_idx, stats)

        for layer_idx in window:
            layer = text.layers[layer_idx]
            hidden = [
                _run_layer(layer, h, mask, ids)
                for h, mask, ids in zip(hidden, masks, input_ids_list)
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
    plan["note"] = (
        "layers 0..num_hash_layers-1 route via a fixed token-hash table, not "
        "learned routing -- their saliency scores are real but reflect a "
        "static assignment, not a tunable preference."
    )
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
