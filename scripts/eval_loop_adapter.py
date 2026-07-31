"""Score a latent-looping adapter on the on-policy corrections eval.

Wraps eval_gemma4_corrections.py's scoring for adapters produced by
reap_stream/lora_loop_gemma4.py, which saves a raw safetensors of trainable
params rather than an mlx-style adapter directory with adapter_config.json --
so mlx_vlm.load(adapter_path=...) cannot read them. This rebuilds the same
LoRA wrapping the trainer used, then loads the weights the same way the
trainer's --resume-adapter-file path does.

The --lora-rank MUST match the rank the adapter was trained at, or the shapes
won't load.

Usage:
    .venv/bin/python scripts/eval_loop_adapter.py \
        --model ~/.lmstudio/models/mlx-community/gemma-4-12B-it-bf16 \
        --adapter-file adapters/stage0_final.safetensors \
        --lora-rank 128 \
        --out artifacts/corrections_stage0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_lm.tuner.lora import LoRALinear

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_gemma4_corrections import DATASETS, load_rows, normalize


def _text_lm(model):
    return getattr(model, "language_model", None) or model


def wrap_and_load(model, adapter_file, rank, scale, unfreeze_first_layers=0):
    text_model = _text_lm(model).model
    n = 0
    for li, layer in enumerate(text_model.layers):
        if li < unfreeze_first_layers:
            continue
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            base = getattr(layer.self_attn, name, None)
            if base is not None and not isinstance(base, LoRALinear):
                setattr(layer.self_attn, name,
                        LoRALinear.from_base(base, r=rank, scale=scale, dropout=0.0))
                n += 1
        for name in ["gate_proj", "up_proj", "down_proj"]:
            base = getattr(layer.mlp, name, None)
            if base is not None and not isinstance(base, LoRALinear):
                setattr(layer.mlp, name,
                        LoRALinear.from_base(base, r=rank, scale=scale, dropout=0.0))
                n += 1
    loaded = mx.load(adapter_file)
    model.update(tree_unflatten(list(loaded.items())))
    mx.eval(model.parameters())
    print(f"[eval] wrapped {n} layers at rank {rank}, loaded {len(loaded)} tensors", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-file", default=None,
                     help="omit to score the unmodified base model")
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--lora-scale", type=float, default=2.0)
    ap.add_argument("--unfreeze-first-layers", type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n-per-dataset", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"[eval] loading {a.model}", flush=True)
    model, processor = load(a.model, processor_config={"trust_remote_code": True})
    if a.adapter_file:
        print(f"[eval] applying adapter {a.adapter_file}", flush=True)
        model = wrap_and_load(model, a.adapter_file, a.lora_rank, a.lora_scale,
                              a.unfreeze_first_layers)
    config = model.config.__dict__

    per_dataset = {}
    results = []
    for ds_path in a.datasets:
        name = Path(ds_path).parent.name
        rows = load_rows(ds_path, a.n_per_dataset)
        n_correct = 0
        for i, r in enumerate(rows):
            msgs = r["messages"]
            user_msg = next(m for m in msgs if m["role"] == "user")
            expected = next(m for m in msgs if m["role"] == "assistant")["content"]
            prompt = apply_chat_template(processor, config, [user_msg],
                                         add_generation_prompt=True)
            out = generate(model, processor, prompt, image=None,
                           max_tokens=a.max_tokens, verbose=False)
            text = out.text if hasattr(out, "text") else str(out)
            pred, exp = normalize(text), normalize(expected)
            ok = pred == exp
            n_correct += ok
            results.append({"dataset": name, "expected": exp,
                            "predicted": pred[:300], "correct": ok})
            if (i + 1) % 25 == 0:
                print(f"  [{name}] {i+1}/{len(rows)} acc={n_correct/(i+1):.3f}", flush=True)
        acc = n_correct / len(rows) if rows else 0.0
        per_dataset[name] = {"n": len(rows), "accuracy": acc}
        print(f"[eval] {name}: {n_correct}/{len(rows)} = {acc:.3f}", flush=True)

    total_n = sum(d["n"] for d in per_dataset.values())
    total_correct = sum(d["n"] * d["accuracy"] for d in per_dataset.values())
    summary = {
        "label": a.label,
        "model": a.model,
        "adapter_file": a.adapter_file,
        "lora_rank": a.lora_rank,
        "per_dataset": per_dataset,
        "overall_accuracy": total_correct / total_n if total_n else 0.0,
        "overall_n": total_n,
        "results": results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(a.out, "w"), indent=2)
        print(f"[eval] wrote -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
