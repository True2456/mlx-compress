#!/usr/bin/env python3
"""
PyTorch MoE Weight Slicer for Step-3.7-Flash Architecture

Applies pruning plans by slicing out removed expert tensors from gate_proj, up_proj, down_proj
and updating the router biases/weights to route exclusively to kept experts.
"""
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any

def apply_step3p7_pytorch_plan(model: nn.Module, plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies an expert pruning plan to a PyTorch Step 3.7 Flash model in-place.
    """
    plan_dict = plan.get("plan", plan)
    num_moe_layers_modified = 0
    total_experts_removed = 0
    expert_counts_per_layer = {}

    print("✂️ Applying Step 3.7 PyTorch Expert Pruning Slices...")

    for layer_idx_str, layer_plan in plan_dict.items():
        layer_idx = int(layer_idx_str)
        prune_indices = layer_plan.get("prune_indices", [])
        if not prune_indices:
            continue

        layer = model.model.layers[layer_idx]
        moe_block = getattr(layer, "block_sparse_moe", getattr(layer, "moe", None))
        if moe_block is None:
            continue

        # Get list of kept expert indices (0..287 except prune_indices)
        all_experts = set(range(288))
        kept_indices = sorted(list(all_experts - set(prune_indices)))
        kept_tensor = torch.tensor(kept_indices, dtype=torch.long)

        # Slice experts module if structured as nn.ModuleList or 3D tensor
        experts = getattr(moe_block, "experts", None)
        if isinstance(experts, nn.ModuleList):
            # Keep only modules at kept_indices
            new_experts = nn.ModuleList([experts[i] for i in kept_indices])
            moe_block.experts = new_experts
        elif hasattr(experts, "gate_proj"): # Combined 3D weight tensor
            with torch.no_grad():
                experts.gate_proj.weight = nn.Parameter(experts.gate_proj.weight[kept_tensor])
                experts.up_proj.weight = nn.Parameter(experts.up_proj.weight[kept_tensor])
                experts.down_proj.weight = nn.Parameter(experts.down_proj.weight[kept_tensor])

        # Slice router weights/biases
        router = getattr(moe_block, "gate", getattr(moe_block, "router", None))
        if router is not None:
            with torch.no_grad():
                if hasattr(router, "weight") and router.weight is not None:
                    router.weight = nn.Parameter(router.weight[kept_tensor])
                if hasattr(router, "bias") and router.bias is not None:
                    router.bias = nn.Parameter(router.bias[kept_tensor])

        num_moe_layers_modified += 1
        total_experts_removed += len(prune_indices)
        expert_counts_per_layer[layer_idx_str] = len(kept_indices)

    summary = {
        "moe_layers_modified": num_moe_layers_modified,
        "total_experts_removed": total_experts_removed,
        "experts_kept_per_layer": expert_counts_per_layer
    }

    print(f"✅ Pruning Slices Applied: Modified {num_moe_layers_modified} MoE layers. Removed {total_experts_removed} total expert weights.")
    return summary
