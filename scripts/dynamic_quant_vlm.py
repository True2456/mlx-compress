"""Dynamic (sensitivity-weighted) quantization for mlx_vlm models.

mlx_lm has a real implementation (mlx_lm/quant/dynamic_quant.py) but can't
load this gemma4_unified checkpoint -- same wall hit with DWQ and fusing.
This ports the same algorithm (verified by reading the real mlx_lm source)
to run through mlx_vlm instead:

  1. Estimate per-layer sensitivity: for each quantizable layer, compute a
     gradient-weighted score of how much quantizing *that layer* to low-bits
     (vs. high-bits) would move the KL-divergence loss against the
     full-precision teacher. First-order Taylor approximation of the
     quantization damage, per layer.
  2. Binary-search a sensitivity threshold so that "protect layers above
     threshold with high-bits, quantize everything else to low-bits" lands
     on the target average bits-per-weight.
  3. Apply the resulting mixed-precision quantization and save.

Multimodal modules (vision_embedder/embed_vision/embed_audio) are always
fully excluded from quantization -- same fix as dwq_vlm.py, same reason:
skip_multimodal_module's pattern list doesn't match this model's actual
naming, and including them corrupted generation in the DWQ smoke test.

Usage:
    .venv/bin/python scripts/dynamic_quant_vlm.py \
        --model /tmp/gemma4-12b-dequant-clean \
        --data data/lora_gemma4/train_no_text_only.jsonl \
        --num-samples 200 --low-bits 2 --high-bits 4 --target-bpw 2.4 \
        --save-path ~/.lmstudio/models/truemod/gemma-4-12b-frontierdistill-orpo-theory-dynamic2bit
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import random
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tqdm import tqdm

import mlx_vlm
import mlx_vlm.utils as u
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.quant_utils import quantize_model
from mlx_vlm.utils import skip_multimodal_module

from mlx_lm.tuner.losses import kl_div_loss

GEMMA4_MULTIMODAL_PREFIXES = ("vision_embedder", "embed_vision", "embed_audio")


def _to_plain(v):
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _to_plain(dataclasses.asdict(v))
    if isinstance(v, dict):
        return {k: _to_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_plain(x) for x in v]
    return v


def is_multimodal(path):
    return skip_multimodal_module(path) or any(
        path.startswith(p) for p in GEMMA4_MULTIMODAL_PREFIXES
    )


def load_calibration(data_path, processor, config, n_samples, max_seq_length, seed):
    rows = [json.loads(l) for l in open(data_path)]
    random.Random(seed).shuffle(rows)
    examples = []
    for r in rows:
        if len(examples) >= n_samples:
            break
        prompt = apply_chat_template(
            processor, config, r["messages"], tools=r.get("tools"), add_generation_prompt=False
        )
        ids = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) < 8:
            continue
        examples.append(mx.array(ids[:max_seq_length])[None])
    return examples


def get_logits(model, tokens, model_type):
    attn = None if model_type == "gemma4_unified" else mx.ones_like(tokens)
    out = model(tokens[:, :-1], None, attn)
    return out.logits.astype(mx.float32)


def compute_bits_per_weight(model):
    total_bits = 0
    total_params = 0
    for _, m in model.named_modules():
        if hasattr(m, "bits") and hasattr(m, "scales"):
            n = m.scales.size * m.group_size
            total_bits += n * m.bits
            total_params += n
        elif hasattr(m, "weight") and isinstance(m.weight, mx.array):
            n = m.weight.size
            total_bits += n * 16
            total_params += n
    return total_bits / max(total_params, 1)


def estimate_sensitivities(model, data, low_bits, low_group_size, high_bits, high_group_size, model_type):
    def qdq(w, bits, group_size):
        w, s, b = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(w, scales=s, biases=b, bits=bits, group_size=group_size)

    lm = model.language_model
    layers = tree_flatten(lm.leaf_modules(), is_leaf=nn.Module.is_module)
    layers = {k: l for k, l in layers if hasattr(l, "to_quantized") and not is_multimodal(k)}
    print(f"[dynq] {len(layers)} quantizable text-decoder layers considered for sensitivity", flush=True)

    q_lm = copy.deepcopy(lm)
    q_layers = copy.deepcopy(layers)
    for l in q_layers.values():
        l.weight = qdq(l.weight, low_bits, low_group_size)
        l.freeze()
        l.unfreeze(keys=["weight"])
    q_lm.freeze()
    q_lm.update_modules(tree_unflatten(list(q_layers.items())))

    def loss_fn(tokens, targets):
        student_out = q_lm(tokens[:, :-1], None, None if model_type == "gemma4_unified" else mx.ones_like(tokens[:, :-1]))
        return kl_div_loss(student_out.logits.astype(mx.float32), targets).mean()

    grad_accum = tree_map(lambda x: mx.zeros(x.shape, dtype=mx.float32), q_lm.trainable_parameters())
    for tokens in tqdm(data, desc="[dynq] estimating sensitivities"):
        targets = mx.stop_gradient(get_logits(model, tokens, model_type))
        mx.eval(targets)
        _, grads = nn.value_and_grad(q_lm, loss_fn)(tokens, targets)
        grad_accum = tree_map(lambda x, y: x + y, grad_accum, grads)
        mx.eval(grad_accum)

    def compute_sensitivity(gradient, low_q_weight, original_weight):
        gradient = gradient / len(data)
        high_q_weight = qdq(original_weight, high_bits, high_group_size)
        param_size = original_weight.size / 1e6
        alignment = (gradient * (low_q_weight - high_q_weight)).sum()
        return alignment / param_size

    sens = tree_map(compute_sensitivity, grad_accum, q_lm.parameters(), lm.parameters())
    mx.eval(sens)
    sens = [(k[: -len(".weight")], s.item()) for k, s in tree_flatten(sens) if k.endswith(".weight")]
    return sens


def estimate_threshold(lm, sensitivities, target_bpw, low_bits, low_group_size, high_bits, high_group_size):
    def predicate(p, m, high_threshold):
        if not hasattr(m, "to_quantized") or is_multimodal(p):
            return False
        if sensitivities.get(p, -1e9) > high_threshold:
            return {"bits": high_bits, "group_size": high_group_size}
        return True

    sens_vals = list(sensitivities.values())
    min_t, max_t = min(sens_vals), max(sens_vals)
    tol = 1e-3 * (max_t - min_t) if max_t > min_t else 1e-6
    while (max_t - min_t) > tol:
        mid = (max_t + min_t) / 2
        q_lm = copy.deepcopy(lm)
        nn.quantize(q_lm, group_size=low_group_size, bits=low_bits, class_predicate=lambda p, m: predicate(p, m, mid))
        bpw = compute_bits_per_weight(q_lm)
        del q_lm
        if bpw > target_bpw:
            min_t = mid
        else:
            max_t = mid
    return (max_t + min_t) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/lora_gemma4/train_no_text_only.jsonl")
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--target-bpw", type=float, default=2.4)
    ap.add_argument("--low-bits", type=int, default=2)
    ap.add_argument("--low-group-size", type=int, default=64)
    ap.add_argument("--high-bits", type=int, default=4)
    ap.add_argument("--high-group-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()

    random.seed(a.seed)
    mx.random.seed(a.seed)

    model_path = str(Path(a.model).expanduser())
    print(f"[dynq] loading teacher {model_path}", flush=True)
    model, processor = mlx_vlm.load(model_path, processor_config={"trust_remote_code": True})
    config = model.config.__dict__
    model_type = config.get("model_type")

    print(f"[dynq] loading calibration data from {a.data}", flush=True)
    data = load_calibration(a.data, processor, config, a.num_samples, a.max_seq_length, a.seed)
    print(f"[dynq] {len(data)} calibration examples", flush=True)

    sensitivities = estimate_sensitivities(
        model, data, a.low_bits, a.low_group_size, a.high_bits, a.high_group_size, model_type
    )
    sensitivities = dict(sensitivities)

    print(f"[dynq] searching threshold for target_bpw={a.target_bpw}", flush=True)
    threshold = estimate_threshold(
        model.language_model, sensitivities, a.target_bpw,
        a.low_bits, a.low_group_size, a.high_bits, a.high_group_size,
    )
    print(f"[dynq] threshold: {threshold:.6f}", flush=True)

    def quant_predicate(p, m):
        if not hasattr(m, "to_quantized") or is_multimodal(p):
            return False
        if sensitivities.get(p, -1e9) > threshold:
            return {"bits": a.high_bits, "group_size": a.high_group_size}
        return True

    print("[dynq] applying final mixed-precision quantization", flush=True)
    model, config = quantize_model(
        model, config, a.low_group_size, a.low_bits, mode="affine", quant_predicate=quant_predicate
    )
    final_bpw = compute_bits_per_weight(model.language_model)
    print(f"[dynq] final text-decoder bits/weight: {final_bpw:.3f}", flush=True)

    save_path = Path(a.save_path).expanduser()
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"[dynq] saving -> {save_path}", flush=True)
    u.save_weights(save_path, model)
    u.save_config(_to_plain(config), save_path / "config.json")

    src = Path(model_path)
    for f in src.iterdir():
        if f.suffix == ".safetensors" or f.name in ("model.safetensors.index.json", "config.json"):
            continue
        if f.is_file():
            shutil.copy2(f, save_path / f.name)

    print(f"[dynq] done -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
