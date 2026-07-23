#!/usr/bin/env python3
"""
Step-3.7-Flash PyTorch REAP Activation Saliency Collector & Gate Evaluator

Target Architecture: Step3p7ForConditionalGeneration (config.json / text_config)
Config Parameters:
- moe_num_experts: 288
- moe_top_k: 8
- moe_router_activation: "sigmoid"
- moe_router_scaling_factor: 3.0
- use_moe_router_bias: true
- moe_layers_enum: 3..44

Implements REAP Saliency Metric:
S_e = sum_t ( gating_score_e(t) * || hidden_activation_e(t) ||_2 )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any

class Step3p7REAPCollector:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        # Expert saliency score accumulator: layer_idx -> Tensor[288]
        self.saliency_scores: Dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks on Step 3.7 MoE router and expert modules."""
        # Locate text model layers
        text_model = getattr(self.model, "model", getattr(self.model, "language_model", self.model))
        layers = getattr(text_model, "layers", [])

        for layer_idx, layer in enumerate(layers):
            # Check if layer is in moe_layers_enum (layers 3..44)
            moe_block = getattr(layer, "block_sparse_moe", getattr(layer, "moe", None))
            if moe_block is not None:
                router = getattr(moe_block, "gate", getattr(moe_block, "router", None))
                if router is not None:
                    self.saliency_scores[layer_idx] = torch.zeros(288, dtype=torch.float32)
                    hook = router.register_forward_hook(self._make_reap_hook(layer_idx, moe_block))
                    self.hooks.append(hook)

        print(f"✅ Step 3.7 REAP Collector: Registered exact REAP saliency hooks on {len(self.hooks)} MoE layers.")

    def _make_reap_hook(self, layer_idx: int, moe_block: nn.Module):
        def hook_fn(module, input_args, output):
            with torch.no_grad():
                # Extract router logits
                if isinstance(output, tuple):
                    router_logits = output[0]
                else:
                    router_logits = output

                # Step 3.7 Sigmoid Router activation with bias & scaling
                router_bias = getattr(module, "bias", None)
                if router_bias is not None:
                    router_logits = router_logits + router_bias

                # Apply Sigmoid activation
                gating_scores = torch.sigmoid(router_logits) * 3.0 # moe_router_scaling_factor = 3.0

                # Top-8 Expert selection
                topk_scores, topk_indices = torch.topk(gating_scores, k=8, dim=-1)

                # Extract input hidden states ||f||_2
                hidden_states = input_args[0]
                hidden_norms = torch.norm(hidden_states, p=2, dim=-1) # [batch, seq]

                # Flatten dimensions
                batch_seq = hidden_norms.shape[0] * hidden_norms.shape[1] if hidden_norms.dim() > 1 else hidden_norms.shape[0]
                topk_indices_flat = topk_indices.view(-1, 8)
                topk_scores_flat = topk_scores.view(-1, 8)
                norms_flat = hidden_norms.view(-1)

                # Accumulate REAP Metric: S_e = sum (gating_score * ||f||_2)
                for i in range(batch_seq):
                    norm_val = norms_flat[i].item()
                    for k in range(8):
                        exp_idx = topk_indices_flat[i, k].item()
                        gate_val = topk_scores_flat[i, k].item()
                        self.saliency_scores[layer_idx][exp_idx] += (gate_val * norm_val)

        return hook_fn

    def get_saliency_dict(self) -> Dict[str, List[float]]:
        """Returns computed layerwise REAP saliency scores for all 288 experts."""
        out = {}
        for layer_idx, scores in self.saliency_scores.items():
            out[str(layer_idx)] = scores.cpu().tolist()
        return out

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
