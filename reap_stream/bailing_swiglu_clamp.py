# SPDX-License-Identifier: Apache-2.0
"""Ling's trained per-layer SwiGLU clamp for ``bailing_hybrid``.

Copied verbatim from the oMLX app patch
(/Applications/oMLX.app/Contents/Resources/omlx/patches/bailing_swiglu_clamp.py,
see docs/../Documents/Ling-3.0-flash-omlx-findings.md) so the streaming
collector doesn't depend on the app bundle, which an oMLX update can silently
overwrite. Ling-3.0-flash is trained with a clamped SwiGLU on its late layers
and ships the limits in config.json (expert_swiglu_limit_list,
share_expert_swiglu_limit_list); mlx-lm's bailing_hybrid does not implement
it. Measured cost of running unclamped: 88.41% -> 71.34% HumanEval.

Call apply_bailing_swiglu_clamp() once, right after importing
mlx_lm.models.bailing_hybrid and before constructing/loading any model.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LIMIT_ATTR = "_omlx_swiglu_limit"


def clamp_disabled() -> bool:
    return os.environ.get("OMLX_LING_NO_SWIGLU_CLAMP") == "1"


def layer_swiglu_limit(limit_list: Any, layer_idx: int) -> Optional[float]:
    if not limit_list or layer_idx >= len(limit_list):
        return None
    limit = limit_list[layer_idx]
    if limit in (None, 0):
        return None
    return float(limit)


def _install(module: Any) -> None:
    import mlx.core as mx
    import mlx.nn as nn

    if not hasattr(module, "clamped_swiglu"):

        def clamped_swiglu(gate, x, limit):
            return mx.minimum(nn.silu(gate), limit) * mx.clip(x, -limit, limit)

        module.clamped_swiglu = clamped_swiglu

    if not hasattr(module, "ClampedSwiGLU"):

        class ClampedSwiGLU(nn.Module):
            """SwitchGLU-signature activation: __call__(x_up, x_gate).

            mlx-lm's SwitchGLU calls activation(x_up, x_gate) and its stock
            SwiGLU returns swiglu(gate, x), i.e. silu is applied to the gate.
            The clamp must match that or it lands on the wrong tensor.
            """

            def __init__(self, limit: float):
                super().__init__()
                self.limit = float(limit)

            def __call__(self, x, gate):
                return module.clamped_swiglu(gate, x, self.limit)

        module.ClampedSwiGLU = ClampedSwiGLU

    mlp_cls = module.MLP
    if not getattr(mlp_cls.__dict__.get("__call__"), "_omlx_clamp", False):
        stock_swiglu = module.swiglu

        def __call__(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            limit = getattr(self, _LIMIT_ATTR, None)
            if limit:
                return self.down_proj(module.clamped_swiglu(gate, up, limit))
            return self.down_proj(stock_swiglu(gate, up))

        __call__._omlx_clamp = True
        mlp_cls.__call__ = __call__


def bind_limits(module: Any, model: Any, config: Any) -> int:
    """Bind per-layer limits onto a constructed model. Returns paths clamped."""
    if clamp_disabled():
        return 0
    expert_list = getattr(config, "expert_swiglu_limit_list", None)
    shared_list = getattr(config, "share_expert_swiglu_limit_list", None)
    if not expert_list and not shared_list:
        return 0

    clamped = 0
    for idx, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        switch_mlp = getattr(mlp, "switch_mlp", None)
        if switch_mlp is not None:
            limit = layer_swiglu_limit(expert_list, idx)
            if limit:
                switch_mlp.activation = module.ClampedSwiGLU(limit)
                clamped += 1
            shared = getattr(mlp, "shared_experts", None)
            if shared is not None:
                limit = layer_swiglu_limit(shared_list, idx)
                if limit:
                    setattr(shared, _LIMIT_ATTR, limit)
                    clamped += 1
        else:
            limit = layer_swiglu_limit(expert_list, idx)
            if limit:
                setattr(mlp, _LIMIT_ATTR, limit)
                clamped += 1
    return clamped


def ensure_swiglu_clamp(module: Any) -> bool:
    if clamp_disabled():
        logger.warning(
            "Ling SwiGLU clamp disabled via OMLX_LING_NO_SWIGLU_CLAMP; the "
            "model will run unclamped and lose accuracy"
        )
        return False

    if getattr(module, "_omlx_swiglu_clamp_native", False):
        return False
    if hasattr(module, "ClampedSwiGLU") and hasattr(module, "layer_swiglu_limit"):
        module._omlx_swiglu_clamp_native = True
        return False

    if getattr(module, "_omlx_swiglu_clamp_installed", False):
        return False

    _install(module)

    args_cls = module.ModelArgs
    if not getattr(args_cls, "_omlx_clamp_args", False):
        original_from_dict = args_cls.from_dict.__func__

        def patched_from_dict(cls, params):
            args = original_from_dict(cls, params)
            args.expert_swiglu_limit_list = params.get("expert_swiglu_limit_list")
            args.share_expert_swiglu_limit_list = params.get(
                "share_expert_swiglu_limit_list"
            )
            return args

        args_cls.from_dict = classmethod(patched_from_dict)
        args_cls._omlx_clamp_args = True

    model_cls = module.Model
    if not getattr(model_cls, "_omlx_clamp_init", False):
        original_init = model_cls.__init__

        def __init__(self, config):
            original_init(self, config)
            clamped = bind_limits(module, self, config)
            if clamped:
                logger.info(
                    "Ling SwiGLU clamp applied to %d expert paths "
                    "(model is trained with it; mlx-lm's bailing_hybrid "
                    "does not implement it)",
                    clamped,
                )

        model_cls.__init__ = __init__
        model_cls._omlx_clamp_init = True

    module._omlx_swiglu_clamp_installed = True
    return True


def apply_bailing_swiglu_clamp() -> bool:
    """Install Ling's SwiGLU clamp on the live bailing_hybrid module.

    Must be called before mlx_lm.utils.load() constructs the model, since
    the clamp hooks Model.__init__.
    """
    try:
        import importlib

        module = importlib.import_module("mlx_lm.models.bailing_hybrid")
    except Exception:
        logger.debug("mlx_lm.models.bailing_hybrid unavailable; clamp skipped")
        return False
    try:
        return ensure_swiglu_clamp(module)
    except Exception:
        logger.warning(
            "Could not install Ling SwiGLU clamp; the model will run "
            "unclamped and lose accuracy",
            exc_info=True,
        )
        return False


__all__ = [
    "bind_limits",
    "clamp_disabled",
    "ensure_swiglu_clamp",
    "layer_swiglu_limit",
    "apply_bailing_swiglu_clamp",
]
