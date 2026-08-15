# SPDX-License-Identifier: Apache-2.0
"""Ling-3.0-flash's KDA safe-gate clamp, missing from mlx-lm's bailing_hybrid.

config.json for Ling-3.0-flash sets kda_safe_gate=True, kda_lower_bound=-5.0
-- the HF reference implementation (modeling_bailing_moe_v3.py) threads these
into chunk_kda/fused_recurrent_kda (from fla.ops.kda), which use them to
clamp the recurrent decay gate's log-value to [lower_bound, 0) via

    g_log = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))

instead of the unclamped default

    g_log = -exp(A_log) * softplus(a + dt_bias)     # unbounded below

mlx_lm.models.bailing_hybrid.KimiDeltaAttention always uses the unclamped
form (via mlx_lm.models.gated_delta.compute_g) and never reads
kda_safe_gate/kda_lower_bound at all, despite ModelArgs declaring both
fields. See docs/LING3-LONG-CONTEXT-QUANT-FINDINGS.md: this is the leading
hypothesis for a ~13.5K-token catastrophic divergence between the teacher
and every quantized variant tested (5 in a row, all identical, all
independent of bit-width/module/kernel choice) -- consistent with a bug
that's present in BOTH sides of every comparison and only manifests once
accumulated activations drive the unclamped gate into extreme values.

This patches KimiDeltaAttention.__call__ specifically (not the shared
gated_delta.py module, which other bailing_hybrid-style architectures use
too) to compute the clamped gate and call gated_delta_kernel/gated_delta_ops
directly, bypassing gated_delta_update's internal (unclamped) compute_g.

Call apply_kda_safe_gate() once, right after importing
mlx_lm.models.bailing_hybrid and before constructing/loading any model.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def clamp_disabled() -> bool:
    import os
    return os.environ.get("OMLX_LING_NO_KDA_SAFE_GATE") == "1"


def _compute_g_safe(A_log, a, dt_bias, lower_bound):
    import mlx.core as mx
    scale = mx.exp(A_log.astype(mx.float32))
    g_log = lower_bound * mx.sigmoid(scale * (a + dt_bias))
    return mx.exp(g_log)


def apply_kda_safe_gate() -> bool:
    """Patch mlx_lm.models.bailing_hybrid.KimiDeltaAttention.__call__ to use
    Ling-3.0-flash's config-specified safe gate clamp. Idempotent."""
    if clamp_disabled():
        logger.warning(
            "KDA safe-gate clamp disabled via OMLX_LING_NO_KDA_SAFE_GATE; "
            "the model will run with the unclamped (possibly unstable at "
            "long context) decay gate"
        )
        return False
    try:
        import importlib
        bh = importlib.import_module("mlx_lm.models.bailing_hybrid")
    except Exception:
        logger.debug("mlx_lm.models.bailing_hybrid unavailable; kda safe-gate skipped")
        return False

    if getattr(bh.KimiDeltaAttention.__call__, "_kda_safe_gate_patched", False):
        return True

    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops

    orig_init = bh.KimiDeltaAttention.__init__

    def __init__(self, args):
        orig_init(self, args)
        self.kda_safe_gate = bool(getattr(args, "kda_safe_gate", False))
        self.kda_lower_bound = getattr(args, "kda_lower_bound", None)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, recurrent_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = recurrent_state = None
            lengths = None

        if q_state is None:
            state = mx.zeros((batch, self.conv_kernel - 1, self.projection_dim), dtype=dtype)
            q_state = k_state = v_state = state

        q, q_state = self.q_conv1d(self.q_proj(x), q_state, mask, lengths)
        k, k_state = self.k_conv1d(self.k_proj(x), k_state, mask, lengths)
        v, v_state = self.v_conv1d(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q.reshape(batch, length, self.num_heads, self.head_dim)
        k = k.reshape(batch, length, self.num_heads, self.head_dim)
        v = v.reshape(batch, length, self.num_heads, self.head_dim)

        q = (self.scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.scale * mx.fast.rms_norm(k, None, 1e-6)

        if self.no_kda_lora:
            decay_logits = self.f_proj(x)
        else:
            decay_logits = self.f_b_proj(self.f_a_proj(x))
        decay_logits = decay_logits.reshape(batch, length, self.num_heads, self.head_dim)
        beta_logits = self.b_proj(x).reshape(batch, length, self.num_heads)
        beta = mx.sigmoid(beta_logits)

        if self.kda_safe_gate and self.kda_lower_bound is not None:
            g = _compute_g_safe(
                self.A_log.reshape(self.num_heads, 1),
                decay_logits,
                self.dt_bias.reshape(self.num_heads, self.head_dim),
                self.kda_lower_bound,
            )
        else:
            from mlx_lm.models.gated_delta import compute_g
            g = compute_g(
                self.A_log.reshape(self.num_heads, 1),
                decay_logits,
                self.dt_bias.reshape(self.num_heads, self.head_dim),
            )

        if recurrent_state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            recurrent_state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

        use_kernel = not self.training
        if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
            output, recurrent_state = gated_delta_ops(q, k, v, g, beta, recurrent_state, mask)
        else:
            output, recurrent_state = gated_delta_kernel(q, k, v, g, beta, recurrent_state, mask)

        if cache is not None:
            cache[3] = recurrent_state
            cache.advance(length)

        if self.no_kda_lora:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(batch, length, self.num_heads, self.head_dim)
        output = self.o_norm(output) * mx.sigmoid(gate)
        return self.o_proj(output.reshape(batch, length, -1))

    __call__._kda_safe_gate_patched = True
    bh.KimiDeltaAttention.__init__ = __init__
    bh.KimiDeltaAttention.__call__ = __call__
    logger.info("KDA safe-gate clamp installed on bailing_hybrid.KimiDeltaAttention")
    return True


__all__ = ["apply_kda_safe_gate", "clamp_disabled"]
