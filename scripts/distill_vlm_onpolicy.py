"""On-policy VLM knowledge distillation: local teacher scores the student's OWN
rollouts, and the student is trained with a token-level KL loss against the
teacher's distribution over exactly the tokens the student produced -- not a
static teacher-generated dataset. Full fine-tune (not LoRA, per instruction):
both full models must be resident in memory simultaneously.

Assumes teacher and student share a tokenizer/processor (true for same-family
distillation, e.g. a REAP-pruned or otherwise smaller checkpoint of the same
model -- matches the build_student*.py naming already in this repo). If they
don't share a vocab, teacher logits can't be compared token-for-token against
student rollouts and this script does not support that case.

Batch size is fixed at 1: rollouts have different lengths and (optionally)
different images per row, so batching would require padding/masking logic
this draft intentionally skips. Loop-level gradient accumulation can be added
if per-step noise turns out to matter.

Usage:
    .venv/bin/python scripts/distill_vlm_onpolicy.py -c artifacts/distill/onpolicy_vlm.yaml

Example config:
    teacher_model: mlx-community/Step-3.7-Flash-bf16
    student_model: artifacts/students/step37_reap15
    data: artifacts/distill/prompts.jsonl   # {"messages": [...], "images": ["a.png"]}
    output_dir: artifacts/distill/onpolicy_ckpt
    iters: 2000
    learning_rate: 1e-6
    max_new_tokens: 256
    gen_temperature: 0.7
    kd_temperature: 1.0
    steps_per_report: 10
    save_every: 200
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import cycle
from pathlib import Path

import numpy as np
import yaml

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer.datasets import NATIVE_PREPROCESS_MODELS
from mlx_vlm.trainer.utils import Colors
from mlx_vlm.utils import (
    load,
    prepare_inputs,
    process_inputs_with_fallback,
    save_weights,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prompt_inputs(processor, config, prompt_text, images):
    """Tokenize a prompt (+ optional images) the same way VisionDataset does,
    so prompt_ids/pixel_values line up with what the model was trained to expect."""
    model_type = config.get("model_type")
    inputs = None
    if model_type in NATIVE_PREPROCESS_MODELS:
        try:
            inputs = process_inputs_with_fallback(
                processor=processor,
                prompts=[prompt_text],
                images=images if images else None,
                audio=None,
                add_special_tokens=False,
            )
            if "images" in inputs and "pixel_values" not in inputs:
                inputs["pixel_values"] = inputs.pop("images")
        except Exception:
            inputs = None

    if inputs is None:
        inputs = prepare_inputs(
            processor=processor,
            images=images if images else None,
            audio=None,
            prompts=[prompt_text],
            image_token_index=config.get("image_token_index"),
            resize_shape=None,
        )
    return inputs


def _align_logits_with_targets(logits, targets):
    if logits.shape[1] < targets.shape[1]:
        pad_length = targets.shape[1] - logits.shape[1]
        pad_width = ((0, 0), (0, pad_length), (0, 0))
        return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
    if logits.shape[1] > targets.shape[1]:
        return logits[:, -targets.shape[1] :, :]
    return logits


def _forward_logits(model, model_config, shifted_input_ids, pixel_values, shifted_attention_mask):
    model_type = (
        model_config.get("model_type")
        if isinstance(model_config, dict)
        else getattr(model_config, "model_type", None)
    )
    # gemma4_unified crashes on a real attention_mask under grad transforms --
    # same guard already proven necessary in orpo_vlm_local.py's patched get_logps.
    attention_mask = None if model_type == "gemma4_unified" else shifted_attention_mask
    outputs = model(shifted_input_ids, pixel_values, attention_mask)
    return outputs.logits.astype(mx.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    logger.info(f"{Colors.HEADER}Loading teacher from {cfg['teacher_model']}{Colors.ENDC}")
    model_t, processor = load(cfg["teacher_model"], processor_config={"trust_remote_code": True})
    logger.info(f"{Colors.HEADER}Loading student from {cfg['student_model']}{Colors.ENDC}")
    model_s, _ = load(cfg["student_model"], processor_config={"trust_remote_code": True})

    teacher_config = model_t.config.__dict__ if hasattr(model_t.config, "__dict__") else model_t.config
    student_config = model_s.config.__dict__ if hasattr(model_s.config, "__dict__") else model_s.config

    model_t.eval()
    model_s.train()

    rows = _load_jsonl(cfg["data"])
    logger.info(f"{Colors.OKBLUE}Loaded {len(rows)} prompts from {cfg['data']}{Colors.ENDC}")

    optimizer = optim.Adam(learning_rate=cfg["learning_rate"])
    T = cfg.get("kd_temperature", 1.0)
    max_new_tokens = cfg.get("max_new_tokens", 256)
    gen_temperature = cfg.get("gen_temperature", 0.7)
    steps_per_report = cfg.get("steps_per_report", 10)
    save_every = cfg.get("save_every", 200)
    iters = cfg["iters"]
    output_dir = Path(cfg["output_dir"])

    def loss_fn(model, shifted_input_ids, pixel_values, shifted_attention_mask, targets, completion_mask, teacher_logp, teacher_p):
        s_logits = _forward_logits(model, student_config, shifted_input_ids, pixel_values, shifted_attention_mask)
        s_logits = _align_logits_with_targets(s_logits, targets)
        student_logp = nn.log_softmax(s_logits / T, axis=-1)
        kl = (teacher_p * (teacher_logp - student_logp)).sum(-1)
        mask_f = completion_mask.astype(kl.dtype)
        denom = mx.maximum(mask_f.sum(), 1)
        return (kl * mask_f).sum() / denom

    running_loss = 0.0
    reported = 0

    for it, row in zip(range(1, iters + 1), cycle(rows)):
        conversation = row["messages"]
        image_paths = row.get("images") or []
        num_images = len(image_paths)

        prompt_text = apply_chat_template(
            processor, teacher_config, conversation, num_images=num_images, add_generation_prompt=True
        )

        prompt_inputs = _prompt_inputs(processor, teacher_config, prompt_text, image_paths)
        prompt_ids = np.array(prompt_inputs["input_ids"]).reshape(1, -1)
        pixel_values_np = prompt_inputs.get("pixel_values")
        prompt_len = prompt_ids.shape[1]

        gen_token_ids = []
        for response in stream_generate(
            model_s, processor, prompt_text,
            image=image_paths or None,
            max_tokens=max_new_tokens,
            temperature=gen_temperature,
        ):
            if response.token is None:
                continue
            gen_token_ids.append(response.token)
            if response.finish_reason is not None:
                break

        if not gen_token_ids:
            logger.warning(f"{Colors.WARNING}Empty rollout at step {it}, skipping{Colors.ENDC}")
            continue

        full_ids_np = np.concatenate(
            [prompt_ids, np.array(gen_token_ids, dtype=np.int64)[None, :]], axis=1
        )
        full_ids = mx.array(full_ids_np)
        attention_mask = mx.ones_like(full_ids)
        pixel_values = mx.array(pixel_values_np) if pixel_values_np is not None else None

        shifted_input_ids = full_ids[:, :-1]
        shifted_attention_mask = attention_mask[:, :-1]
        targets = full_ids[:, 1:]

        completion_start = prompt_len - 1
        steps = mx.arange(shifted_input_ids.shape[1])[None, :]
        completion_mask = steps >= completion_start

        t_logits = _forward_logits(model_t, teacher_config, shifted_input_ids, pixel_values, shifted_attention_mask)
        t_logits = _align_logits_with_targets(t_logits, targets)
        teacher_logp = mx.stop_gradient(nn.log_softmax(t_logits / T, axis=-1))
        teacher_p = mx.stop_gradient(mx.exp(teacher_logp))

        loss, grads = nn.value_and_grad(model_s, loss_fn)(
            model_s, shifted_input_ids, pixel_values, shifted_attention_mask,
            targets, completion_mask, teacher_logp, teacher_p,
        )
        optimizer.update(model_s, grads)
        mx.eval(model_s.parameters(), optimizer.state)

        running_loss += loss.item()
        reported += 1

        if it % steps_per_report == 0:
            logger.info(
                f"{Colors.OKGREEN}step {it}/{iters}  kd_loss {running_loss / reported:.4f}  "
                f"completion_tokens {len(gen_token_ids)}{Colors.ENDC}"
            )
            running_loss = 0.0
            reported = 0

        if it % save_every == 0 or it == iters:
            ckpt_dir = output_dir / f"step_{it}"
            logger.info(f"{Colors.UNDERLINE}Saving checkpoint to {ckpt_dir}{Colors.ENDC}")
            save_weights(ckpt_dir, model_s)


if __name__ == "__main__":
    main()
