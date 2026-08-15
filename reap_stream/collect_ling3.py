"""Streaming REAP + DWQ-target collector for Ling-3.0-flash (mlx_lm bailing_hybrid).

Same windowed-streaming pattern as collect_step3p7.py (embed once, then run
decoder blocks in windows of `layers_at_once`, freeing each window's weights
before the next), ported to bailing_hybrid's module tree:

  - text model:     model.model  (LanguageModel)
  - embed:          text.word_embeddings   (not embed_tokens)
  - decoder layer:  DecoderLayer(attention=MLA|KDA, mlp=SparseMoeBlock|MLP)
  - MoE block:      layer.mlp.{gate, switch_mlp, shared_experts}
  - two mask kinds: create_attention_mask for MLA (layer.is_global), plus
    create_ssm_mask for KDA layers (recurrent linear attention)

Applies bailing_swiglu_clamp before loading -- mlx-lm's stock bailing_hybrid
is missing Ling's trained per-layer SwiGLU clamp on layers 34-41, which is a
measured 17pp HumanEval regression if left unclamped (see
Documents/Ling-3.0-flash-omlx-findings.md). Both REAP saliency and DWQ
teacher logits would otherwise be computed on the wrong activations.
"""
from __future__ import annotations

import gc
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from .bailing_swiglu_clamp import apply_bailing_swiglu_clamp
from .saliency import LayerSaliency

# Release MLX's buffer cache every N per-prompt forwards -- see collect_step3p7.py
_CACHE_EVERY = 200


def _text_model(model):
    return model.model


def _moe_layer_ids(text) -> list[int]:
    import mlx_lm.models.bailing_hybrid as bh

    return [i for i, layer in enumerate(text.layers)
            if isinstance(layer.mlp, bh.SparseMoeBlock)]


def _num_experts(text) -> int:
    import mlx_lm.models.bailing_hybrid as bh

    for layer in text.layers:
        if isinstance(layer.mlp, bh.SparseMoeBlock):
            return int(layer.mlp.gate.gate_proj.weight.shape[0])
    raise RuntimeError("no MoE layers found")


class _MoEProbe(nn.Module):
    """Wrap bailing_hybrid's SparseMoeBlock to accumulate REAP saliency
    without changing the forward math."""

    def __init__(self, inner: nn.Module, layer_idx: int, stats: dict[int, LayerSaliency]):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self._stats = stats

    def __call__(self, x):
        indices, weights = self.inner.gate(x)
        y = self.inner.switch_mlp(x, indices)          # [..., top_k, hidden]
        norms = mx.sqrt((y.astype(mx.float32) ** 2).sum(axis=-1) + 1e-12)
        weights32 = weights.astype(mx.float32)
        mx.eval(indices, weights32, norms)
        ids = np.array(indices, dtype=np.int64).reshape(-1, indices.shape[-1])
        gates = np.array(weights32, dtype=np.float64).reshape(-1, weights32.shape[-1])
        nrm = np.array(norms, dtype=np.float64).reshape(-1, norms.shape[-1])
        self._stats[self.layer_idx].update(ids, gates, nrm)
        output = (y * weights[..., None]).sum(axis=-2)
        if self.inner.shared_experts is not None:
            output = output + self.inner.shared_experts(x)
        return output


class _DropLayer(nn.Module):
    def __call__(self, h, *args, **kwargs):
        raise RuntimeError("freed layer was invoked - layerwise collector bug")


def _free_layer(text, layer_idx: int) -> None:
    text.layers[layer_idx] = _DropLayer()


def _run_layer(layer, h, cache=None):
    if layer.is_global:
        mask = create_attention_mask(h, cache, return_array=True)
    else:
        mask = create_ssm_mask(h, cache)
    h_out = layer(h, mask=mask, cache=cache)
    mx.eval(h_out)
    return h_out


def load_lazy(model_path: str):
    """Load bailing_hybrid lazily with Ling's SwiGLU clamp installed first."""
    apply_bailing_swiglu_clamp()
    model, tokenizer = load(model_path, lazy=True)
    return model, tokenizer


def _truncate(tokens: list[int], max_tokens: int, mode: str) -> list[int]:
    """See docs/step37-reap-dwq-summary.md sec 6b: `head` silently drops the
    ASSISTANT answer on long agentic/tool prompts. `headtail` was the
    validated fix there; kept as the default policy here too."""
    if len(tokens) <= max_tokens:
        return list(tokens)
    if mode == "tail":
        return list(tokens[-max_tokens:])
    if mode == "headtail":
        h = max_tokens // 2
        t = max_tokens - h
        return list(tokens[:h]) + list(tokens[-t:])
    return list(tokens[:max_tokens])  # head


def _tokenize_prompts(tokenizer, prompts: list[str], max_tokens: int,
                      truncation: str = "headtail") -> list[list[int]]:
    batches: list[list[int]] = []
    for p in prompts:
        try:
            text_in = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text_in = p
        tokens = tokenizer.encode(text_in)
        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]
        batches.append(_truncate(list(tokens), max_tokens, truncation))
    return batches


def inspect_model(model_path: str) -> dict:
    """Load lazy and report MoE layout (no forward). Mirrors collect_step3p7's
    inspect_model for a quick sanity check before a real streaming run."""
    model, tokenizer = load_lazy(model_path)
    text = _text_model(model)
    moe_ids = _moe_layer_ids(text)
    n_experts = _num_experts(text) if moe_ids else 0
    global_ids = [i for i, l in enumerate(text.layers) if l.is_global]
    info = {
        "arch": "bailing_hybrid",
        "n_layers": len(text.layers),
        "moe_layers": moe_ids,
        "n_moe_layers": len(moe_ids),
        "moe_num_experts": n_experts,
        "global_attention_layers": global_ids,
        "kda_layers": [i for i in range(len(text.layers)) if i not in global_ids],
    }
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return info


def run_window_smoke(model_path: str, prompt: str, n_layers: int, layers_at_once: int = 2):
    """Stream the first n_layers decoder blocks on one prompt. Smoke test only
    -- confirms the windowed forward doesn't crash and memory is freed between
    windows, before committing to a full run."""
    model, tokenizer = load_lazy(model_path)
    text = _text_model(model)

    tokens = tokenizer.encode(prompt)
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    h = text.word_embeddings(mx.array(tokens)[None])
    mx.eval(h)

    for w0 in range(0, n_layers, layers_at_once):
        window = range(w0, min(w0 + layers_at_once, n_layers))
        for li in window:
            layer = text.layers[li]
            h = _run_layer(layer, h)
        for li in window:
            _free_layer(text, li)
        gc.collect()
        mx.clear_cache()
        act = mx.get_active_memory() / 1e9
        print(f"[ling3-smoke] blocks {w0}-{window[-1]}/{n_layers - 1} "
              f"active={act:.2f} GB", flush=True)

    print(f"[ling3-smoke] OK, final hidden shape={h.shape}, dtype={h.dtype}", flush=True)
    return h
