"""DWQ (distilled weight quantization) for mlx_vlm models.

mlx_lm has a real DWQ implementation (mlx_lm/quant/dwq.py) but it can't load
this gemma4_unified checkpoint at all -- confirmed earlier this session, not
assumed. This reimplements the same algorithm (verified by reading the real
mlx_lm source, not guessed) against mlx_vlm instead, so vision/audio survive
untouched:

  1. Load the full-precision model as the teacher.
  2. Build a quantized "student" copy via mlx_vlm.quant_utils.quantize_model,
     explicitly skipping multimodal modules from quantization (mirrors
     mlx_vlm.convert's own default predicate) so vision/audio stay at full
     precision.
  3. Unfreeze only the student's scales/biases (not the discrete weight
     codes -- those aren't directly optimizable) for every quantized layer
     with bits < 8.
  4. For each calibration example: get the teacher's top-1024 logits (memory
     trick from the original), compute the student's logits at the same
     positions, minimize KL divergence via Adam on scales/biases only.
  5. Bake the optimized scales/biases back into the model and save.

Calibration data: real agentic trace rows (train_no_text_only.jsonl -- the
action-taking subset, matching the actual deployment distribution), not an
unrelated external dataset -- calibration should match what the model will
actually see, same reasoning we used picking calibration data before.

Usage:
    .venv/bin/python scripts/dwq_vlm.py \
        --model /tmp/gemma4-12b-dequant-clean \
        --data data/lora_gemma4/train_no_text_only.jsonl \
        --num-samples 1000 --bits 2 --group-size 64 \
        --save-path ~/.lmstudio/models/truemod/gemma-4-12b-frontierdistill-orpo-theory-dwq2bit
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from mlx.utils import tree_map
from tqdm import tqdm

import mlx_vlm
import mlx_vlm.utils as u
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.quant_utils import quantize_model
from mlx_vlm.utils import skip_multimodal_module

from mlx_lm.tuner.losses import kl_div_loss

import dataclasses


def _to_plain(v):
    """quantize_model can leave nested config values (e.g. audio_config/
    vision_config) as dataclass instances instead of plain dicts, which
    json.dump chokes on -- confirmed via the real TypeError, not assumed."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _to_plain(dataclasses.asdict(v))
    if isinstance(v, dict):
        return {k: _to_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_plain(x) for x in v]
    return v


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


# skip_multimodal_module's pattern list ("vision_model", "vision_tower", ...)
# doesn't match this model's actual naming -- verified against the real
# safetensors index earlier (vision_embedder, embed_vision, embed_audio),
# not assumed. Without this, patch_dense/embedding_projection get quantized
# to 2-bit with zero calibration and the model emits garbled <audio|>/<image|>
# tokens in plain text generation -- confirmed via the actual smoke test output.
GEMMA4_MULTIMODAL_PREFIXES = ("vision_embedder", "embed_vision", "embed_audio")


def base_quant_predicate(path, module):
    if skip_multimodal_module(path) or any(path.startswith(p) for p in GEMMA4_MULTIMODAL_PREFIXES):
        return False
    return hasattr(module, "to_quantized")


def get_logits(model, tokens, model_type):
    attn = None if model_type == "gemma4_unified" else mx.ones_like(tokens)
    out = model(tokens[:, :-1], None, attn)
    return out.logits.astype(mx.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Full-precision (dequantized) teacher model path")
    ap.add_argument("--data", default="data/lora_gemma4/train_no_text_only.jsonl")
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--num-valid", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=1025)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()

    random.seed(a.seed)
    mx.random.seed(a.seed)

    model_path = str(Path(a.model).expanduser())
    print(f"[dwq] loading teacher {model_path}", flush=True)
    model, processor = mlx_vlm.load(model_path, processor_config={"trust_remote_code": True})
    config = model.config.__dict__
    model_type = config.get("model_type")

    print(f"[dwq] loading calibration data from {a.data}", flush=True)
    all_examples = load_calibration(
        a.data, processor, config, a.num_samples + a.num_valid, a.max_seq_length, a.seed
    )
    train_examples = all_examples[: a.num_samples]
    valid_examples = all_examples[a.num_samples :]
    print(f"[dwq] train: {len(train_examples)}, valid: {len(valid_examples)}", flush=True)

    print(f"[dwq] building {a.bits}-bit quantized student (multimodal modules skipped)", flush=True)
    student = copy.deepcopy(model)
    student, config = quantize_model(
        student, config, a.group_size, a.bits, mode="affine", quant_predicate=base_quant_predicate
    )

    def unfreeze(_, m):
        if hasattr(m, "bits") and hasattr(m, "group_size") and m.mode == "affine" and m.bits < 8:
            m.unfreeze(keys=["scales", "biases"], recurse=False)

    student.freeze()
    student.apply_to_modules(unfreeze)
    print("[dwq] student ready, scales/biases unfrozen", flush=True)

    scale = 1 / a.temperature

    def loss_fn(params, tokens):
        student.update(tree_map(lambda x: x.astype(mx.bfloat16), params))
        student_logits = get_logits(student, tokens, model_type)
        teacher_logits = mx.stop_gradient(get_logits(model, tokens, model_type))
        idx = mx.argpartition(teacher_logits, kth=-1024, axis=-1)[..., -1024:]
        t = mx.take_along_axis(teacher_logits, idx, axis=-1)
        s = mx.take_along_axis(student_logits, idx, axis=-1)
        losses = kl_div_loss(scale * s, scale * t)
        return losses.mean()

    opt = optimizers.Adam(learning_rate=a.learning_rate, bias_correction=True)
    params = tree_map(lambda x: x.astype(mx.float32), student.trainable_parameters())

    def validate(params):
        losses = []
        for tokens in valid_examples:
            l = loss_fn(params, tokens)
            mx.eval(l)
            losses.append(l.item())
        return sum(losses) / len(losses)

    print("[dwq] computing initial validation loss", flush=True)
    initial_valid = valid_loss = validate(params)
    print(f"[dwq] initial valid loss: {initial_valid:.4f}", flush=True)

    tic = time.time()
    total_loss = 0.0
    for it, tokens in enumerate(tqdm(train_examples, desc="dwq")):
        (loss,), grads = mx.value_and_grad(lambda p: (loss_fn(p, tokens),))(params)
        params = opt.apply_gradients(grads, params)
        mx.eval(loss, params)
        total_loss += loss.item()
        if (it + 1) % 50 == 0:
            avg = total_loss / 50
            total_loss = 0.0
            elapsed = time.time() - tic
            print(f"[dwq] it={it+1} avg_loss={avg:.4f} elapsed={elapsed:.1f}s", flush=True)
            tic = time.time()
        if (it + 1) % 250 == 0:
            valid_loss = validate(params)
            print(f"[dwq] it={it+1} valid_loss={valid_loss:.4f}", flush=True)

    valid_loss = validate(params)
    print(f"[dwq] final valid loss: {valid_loss:.4f} (initial: {initial_valid:.4f})", flush=True)
    if valid_loss > initial_valid:
        print("[dwq] WARNING: final validation loss is worse than initial -- quality may be degraded", flush=True)

    student.update(tree_map(lambda x: x.astype(mx.bfloat16), params))

    save_path = Path(a.save_path).expanduser()
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"[dwq] saving -> {save_path}", flush=True)
    u.save_weights(save_path, student)
    u.save_config(_to_plain(config), save_path / "config.json")

    src = Path(model_path)
    import shutil
    for f in src.iterdir():
        if f.suffix == ".safetensors" or f.name in ("model.safetensors.index.json", "config.json"):
            continue
        if f.is_file():
            shutil.copy2(f, save_path / f.name)

    print(f"[dwq] done -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
