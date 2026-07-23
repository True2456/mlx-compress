#!/usr/bin/env python3
"""
Perplexity & Loss Evaluation Module for REAP Pruned Models
Computes exact Cross-Entropy Loss and Perplexity (PPL = exp(loss)) on held-out validation datasets.
"""
import math
import time
import torch
import torch.nn as nn
from typing import List, Dict, Any

def calculate_perplexity(
    model: nn.Module,
    tokenizer: Any,
    val_texts: List[str],
    max_length: int = 2048,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Computes cross-entropy loss and perplexity over a list of validation text sequences.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    print(f"📊 Running Perplexity evaluation over {len(val_texts)} validation sequences...")

    with torch.no_grad():
        for i, text in enumerate(val_texts):
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            input_ids = encodings.input_ids.to(device)
            target_ids = input_ids.clone()

            # Forward pass
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss
            num_tokens = input_ids.size(1)

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if (i + 1) % 50 == 0 or (i + 1) == len(val_texts):
                avg_loss_so_far = total_loss / total_tokens if total_tokens > 0 else 0.0
                ppl_so_far = math.exp(avg_loss_so_far) if avg_loss_so_far < 100 else float('inf')
                print(f"   [{i+1}/{len(val_texts)}] Tokens: {total_tokens} | Loss: {avg_loss_so_far:.4f} | PPL: {ppl_so_far:.4f}")

    elapsed_time = time.time() - start_time
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed_time
    }

def print_perplexity_comparison(results: Dict[str, Dict[str, float]]):
    """
    Prints a formatted comparison table for Base vs Pruned model variants.
    """
    print("\n" + "=" * 70)
    print(f"{'MODEL VARIANT':<25} | {'LOSS':<10} | {'PERPLEXITY (PPL)':<18} | {'PPL Δ vs BASE':<12}")
    print("=" * 70)

    base_ppl = results.get("Base Unpruned", {}).get("perplexity", None)

    for variant, metrics in results.items():
        loss = metrics.get("loss", 0.0)
        ppl = metrics.get("perplexity", 0.0)
        
        if base_ppl and variant != "Base Unpruned":
            delta_pct = ((ppl - base_ppl) / base_ppl) * 100
            delta_str = f"+{delta_pct:.2f}%" if delta_pct >= 0 else f"{delta_pct:.2f}%"
        else:
            delta_str = "BASELINE"

        print(f"{variant:<25} | {loss:<10.4f} | {ppl:<18.4f} | {delta_str:<12}")
    print("=" * 70 + "\n")
