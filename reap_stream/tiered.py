"""Saliency-tiered expert banks: per-expert mixed-precision quantization.

MLX quantizes a SwitchGLU's stacked expert tensor at one bit-width, so
per-expert bits require physically splitting each layer's kept experts into
banks (hot/base/cold) quantized at different widths. TieredSwitchGLU presents
the exact SwitchGLU interface (x, global-position indices) and dispatches each
token-expert pair to its bank via precomputed index maps, so MoE/gate/router
logic is untouched and routing is bit-for-bit identical to the untired model.

Cost: each bank runs the full top-k gather (non-members pointed at slot 0 and
masked out), so expert FLOPs are ~n_banks x. Fine for evals; a production
kernel would partition indices instead.

Deployability note: stock LM Studio resolves step3p7 from its bundled mlx_vlm
package and cannot load this class — tiered models are for local measurement.
`maybe_patch_tiered(model_path)` must be called before mlx_vlm.load() on any
model whose config.json carries "tiered_expert_banks".
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class TieredSwitchGLU(nn.Module):
    def __init__(self, input_dims, hidden_dims, sizes, activation):
        super().__init__()
        from mlx_vlm.models.switch_layers import SwitchGLU

        self.banks = [
            SwitchGLU(input_dims, hidden_dims, n, activation=activation)
            for n in sizes
        ]
        total = sum(sizes)
        self.map_bank = mx.zeros((total,), dtype=mx.int32)
        self.map_slot = mx.zeros((total,), dtype=mx.int32)

    def __call__(self, x, indices) -> mx.array:
        bank = self.map_bank[indices]
        slot = self.map_slot[indices]
        out = None
        for b, glu in enumerate(self.banks):
            idx_b = mx.where(bank == b, slot, mx.zeros_like(slot))
            y = glu(x, idx_b)
            y = mx.where((bank == b)[..., None], y, mx.zeros_like(y))
            out = y if out is None else out + y
        return out


def read_tier_config(model_path) -> dict | None:
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    return cfg.get("tiered_expert_banks")


def maybe_patch_tiered(model_path) -> bool:
    """If model_path is a tiered model, patch step3p7's MoE to build
    TieredSwitchGLU skeletons so load_weights finds matching keys.
    Returns True if the patch is active."""
    tier = read_tier_config(model_path)
    if not tier:
        return False
    import mlx_vlm.models.step3p7.language as lang

    sizes = tier["sizes"]
    orig_init = lang.MoE.__init__
    if getattr(lang.MoE, "_tiered_sizes", None) == sizes:
        return True

    def patched_init(self, config, layer_idx):
        orig_init(self, config, layer_idx)
        activation = self.switch_mlp.activation
        self.switch_mlp = TieredSwitchGLU(
            config.hidden_size, config.moe_intermediate_size, sizes, activation
        )

    if not getattr(lang.MoE.__init__, "_is_tiered_patch", False):
        patched_init._is_tiered_patch = True
        patched_init._orig = orig_init
        lang.MoE.__init__ = patched_init
    lang.MoE._tiered_sizes = sizes
    return True
