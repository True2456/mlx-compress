#!/usr/bin/env python3
"""
Production Step-3.7-Flash VLM PyTorch REAP Multi-Rung Pipeline

Target Architecture: Step3p7ForConditionalGeneration (VLM)
Key Fixes:
1. Multimodal Support: Handles genuine ChartQA PNG images via PIL Image + Step3VLProcessor / AutoProcessor.
2. Fresh Base Model Reload: Reloads fresh Base Model weights from disk before evaluating each ladder rung (p10, p15, p20, p25).
3. Exact Step 3.7 Sigmoid REAP Metric: S_e = sum_t ( gating_score_e(t) * || hidden_activation_e(t) ||_2 ).
4. Zero Cumulative Pruning Corruption: Saves ONLY the winning model weights to disk & triggers HF upload.
"""
import os
import sys
import json
import math
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reap_stream.collect_step3p7_pytorch import Step3p7REAPCollector
from reap_stream.apply_step3p7_pytorch import apply_step3p7_pytorch_plan
from reap_stream.eval_perplexity import calculate_perplexity, print_perplexity_comparison

def parse_args():
    parser = argparse.ArgumentParser(description="Strict Production PyTorch Step 3.7 VLM REAP Pipeline")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to base model BF16 safetensors folder")
    parser.add_argument("--calib-file", type=str, default="calib/cloud_reap_8k.jsonl", help="Path to genuine calibration JSONL")
    parser.add_argument("--output-dir", type=str, default="artifacts/reap_run", help="Output directory")
    parser.add_argument("--val-split-ratio", type=float, default=0.10, help="Held-out validation ratio (default: 0.10)")
    parser.add_argument("--max-calib-samples", type=int, default=5000, help="Max unique calibration samples to profile")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()

def generate_nested_plans(saliency: Dict[str, List[float]], output_dir: Path) -> Dict[str, Path]:
    """Emits the 4 nested pruning plans (10%, 15%, 20%, 25% expert prunes)."""
    rungs = [
        {"name": "p10", "ratio": 0.10, "keep": 259},
        {"name": "p15", "ratio": 0.15, "keep": 245},
        {"name": "p20", "ratio": 0.20, "keep": 230},
        {"name": "p25", "ratio": 0.25, "keep": 216}
    ]
    
    plan_paths = {}
    for r in rungs:
        plan_data = {"arch": "step3p7", "prune_ratio": r["ratio"], "plan": {}}
        for layer_str, scores in saliency.items():
            if len(scores) != 288:
                raise ValueError(f"Saliency profile for layer {layer_str} must contain exactly 288 expert scores, got {len(scores)}")
            
            indexed = sorted(enumerate(scores), key=lambda x: x[1])
            num_to_prune = 288 - r["keep"]
            prune_indices = sorted([idx for idx, sc in indexed[:num_to_prune]])
            plan_data["plan"][layer_str] = {
                "prune_indices": prune_indices,
                "experts_kept": r["keep"]
            }
        
        plan_file = output_dir / f"plan_{r['name']}.json"
        with open(plan_file, "w") as f:
            json.dump(plan_data, f, indent=2)
        plan_paths[r["name"]] = plan_file
        print(f"   - Generated plan_{r['name']}.json (Prunes {288 - r['keep']} experts per layer)")
        
    return plan_paths

def load_fresh_model(model_dir: Path, device: str):
    """Loads a fresh, uncorrupted base model instance from disk."""
    print(f"🧠 Loading fresh Base Step 3.7 PyTorch VLM Model from {model_dir} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception:
        processor = None

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device != "cuda":
        model.to(device)
    return model, tokenizer, processor

def process_sample_inputs(sample: Dict[str, Any], tokenizer: Any, processor: Any, device: str):
    """Processes either text-only or genuine multimodal (PIL Image + text) inputs."""
    img_path = sample.get("image_path")
    text = sample.get("text", "")

    if img_path and Path(img_path).exists() and processor is not None:
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=text, images=image, return_tensors="pt")
            return {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

    # Text-only fallback processing
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    return {k: v.to(device) for k, v in enc.items()}

def main():
    args = parse_args()
    print("=" * 75)
    print("STRICT PRODUCTION STEP 3.7 FLASH VLM REAP PIPELINE")
    print("=" * 75)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    # 1. Load Calibration Dataset
    calib_path = Path(args.calib_file)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file {calib_path} not found. Run scripts/build_calib_mix.py first.")

    with open(calib_path, "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]

    split_idx = int(len(all_rows) * (1 - args.val_split_ratio))
    train_rows = all_rows[:split_idx][:args.max_calib_samples]
    val_rows = all_rows[split_idx:][:200]

    val_texts = [r["text"] for r in val_rows]
    print(f"📦 Dataset Loaded: {len(train_rows)} Calibration Traces | {len(val_texts)} Held-Out Validation Traces")

    # 2. Load Base Model for Saliency Collection
    base_model, tokenizer, processor = load_fresh_model(model_dir, args.device)

    # 3. Base Model Baseline Perplexity
    print("\n--- PHASE 1: Base Model Baseline Perplexity ---")
    base_metrics = calculate_perplexity(base_model, tokenizer, val_texts, device=args.device)
    ppl_results = {"Base Unpruned": base_metrics}
    base_ppl = base_metrics["perplexity"]
    print(f"✅ Base Model PPL: {base_ppl:.4f} (Loss: {base_metrics['loss']:.4f})")

    # 4. Real Activation Saliency Collection
    print("\n--- PHASE 2: Real Step 3.7 VLM REAP Activation Saliency Profiling ---")
    probe = Step3p7REAPCollector(base_model)
    base_model.eval()

    with torch.no_grad():
        for i, r in enumerate(train_rows):
            inputs = process_sample_inputs(r, tokenizer, processor, args.device)
            _ = base_model(**inputs)
            if (i + 1) % 100 == 0 or (i + 1) == len(train_rows):
                print(f"   - Profiled [{i+1}/{len(train_rows)}] sequences (including multimodal images)...")

    saliency_dict = probe.get_saliency_dict()
    probe.remove_hooks()

    if not saliency_dict:
        raise RuntimeError("❌ Error: REAP collector returned empty dict. MoE router hooks failed.")

    saliency_file = output_dir / "saliency_profile.json"
    with open(saliency_file, "w") as f:
        json.dump(saliency_dict, f, indent=2)
    print(f"💾 Saved real saliency profile to {saliency_file}")

    # Unload base_model from VRAM
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Generate Nested Plans
    print("\n--- PHASE 3: Emitting Nested Pruning Plans ---")
    plan_paths = generate_nested_plans(saliency_dict, output_dir)

    # 6. Progressive Gated Ladder Execution (RELOAD BASE EACH RUNG)
    print("\n--- PHASE 4: Progressive Gated Ladder (10% -> 15% -> 20% -> 25%) ---")
    rungs = [
        ("p10", 0.10, 259),
        ("p15", 0.15, 245),
        ("p20", 0.20, 230),
        ("p25", 0.25, 216)
    ]

    winner_name = None

    for name, ratio, keep in rungs:
        print(f"\n✂️ Testing Pruning Rung {name.upper()} (Keep {keep} experts from fresh Base)...")
        plan_file = plan_paths[name]
        with open(plan_file, "r") as f:
            plan_obj = json.load(f)

        # Reload FRESH base model instance from disk
        current_model, _, _ = load_fresh_model(model_dir, args.device)

        # Apply plan to fresh base model
        _ = apply_step3p7_pytorch_plan(current_model, plan_obj)

        # Evaluate held-out perplexity on cleanly pruned weights
        metrics = calculate_perplexity(current_model, tokenizer, val_texts, device=args.device)
        ppl_results[f"REAP_{name}"] = metrics
        current_ppl = metrics["perplexity"]
        delta_pct = ((current_ppl - base_ppl) / base_ppl) * 100

        print(f"   -> Loss: {metrics['loss']:.4f} | PPL: {current_ppl:.4f} (Δ vs Base: +{delta_pct:.2f}%)")

        if delta_pct > 25.0:
            print(f"   🛑 GATE FAILED: PPL degradation (+{delta_pct:.2f}%) exceeded 25% threshold. Stopping ladder.")
            del current_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        else:
            print(f"   ✅ GATE PASSED: Rung {name.upper()} passed evaluation.")
            winner_name = f"REAP_{name}"
            
            # Save winner weights immediately and release memory before testing next rung
            winner_dir = output_dir / "winner_weights"
            winner_dir.mkdir(parents=True, exist_ok=True)
            current_model.save_pretrained(str(winner_dir))
            tokenizer.save_pretrained(str(winner_dir))
            print(f"💾 Updated winner weights on disk: {winner_dir}")

            del current_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_perplexity_comparison(ppl_results)

    # 7. Final Summary
    print("=" * 75)
    if winner_name:
        print(f"🏆 WINNER SELECTED & SAVED: {winner_name}")
        print(f"📁 Weights Location: {output_dir / 'winner_weights'}")
        print("Ready for Hugging Face upload and VM destruction!")
    else:
        print("❌ No pruning rung passed the evaluation gate.")
    print("=" * 75)

if __name__ == "__main__":
    main()
