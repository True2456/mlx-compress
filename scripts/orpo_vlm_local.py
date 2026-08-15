"""Launch mlx_vlm ORPO training against local chosen/rejected JSONL data.

Same reasoning as scripts/lora_vlm_local.py: no stock CLI accepts local files
for preference data either, and PreferenceVisionDataset (unlike VisionDataset)
takes {"chosen": [...], "rejected": [...]} rows directly -- verified against
the real class in trainer/datasets.py before writing this, not assumed.

Usage:
    .venv/bin/python scripts/orpo_vlm_local.py -c artifacts/lora/orpo_stopping.yaml
"""
from __future__ import annotations

import argparse
import logging

import mlx.optimizers as optim
import yaml
from datasets import load_dataset

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_vlm.trainer.orpo_trainer as orpo_trainer
from mlx_vlm.trainer.datasets import PreferenceVisionDataset
from mlx_vlm.trainer.orpo_trainer import ORPOTrainingArgs, train_orpo
from mlx_vlm.trainer.utils import (
    Colors,
    apply_lora_layers,
    find_all_linear_names,
    get_peft_model,
    print_trainable_parameters,
)
from mlx_vlm.utils import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _patched_get_logps(model, batch, train_on_completions=False, assistant_id=77091):
    """mlx_vlm's orpo_trainer.get_logps is missing the gemma4_unified guard that
    sft_trainer.vision_language_loss_fn already has: passing a real attention_mask
    for this model type makes it eagerly .item() inside _should_disable_chunked_prefill,
    which MLX disallows during a gradient transformation. Mirrors the SFT trainer's
    already-working fix (attention_mask=None for gemma4_unified) -- verified against
    the real crash traceback, not guessed."""
    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_size, seq_length = input_ids.shape

    shifted_input_ids = input_ids[:, :-1]
    shifted_attention_mask = attention_mask[:, :-1]
    targets = input_ids[:, 1:]

    kwargs = {
        k: v
        for k, v in batch.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    config = getattr(model, "config", None)
    model_type = config.get("model_type") if isinstance(config, dict) else getattr(config, "model_type", None)
    model_attention_mask = None if model_type == "gemma4_unified" else shifted_attention_mask

    outputs = model(shifted_input_ids, pixel_values, model_attention_mask, **kwargs)
    logits = outputs.logits.astype(mx.float32)

    def align_logits_with_targets(logits, targets):
        if logits.shape[1] < targets.shape[1]:
            pad_length = targets.shape[1] - logits.shape[1]
            pad_width = ((0, 0), (0, pad_length), (0, 0))
            return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
        if logits.shape[1] > targets.shape[1]:
            return logits[:, -targets.shape[1] :, :]
        return logits

    logits = align_logits_with_targets(logits, targets)

    lengths = mx.sum(shifted_attention_mask, axis=1)
    lengths = mx.minimum(lengths, shifted_input_ids.shape[1])
    steps = mx.arange(shifted_input_ids.shape[1])[None, :]
    base_mask = steps < lengths[:, None]

    if train_on_completions:
        assistant_response_index = np.full((batch_size,), -1, dtype=np.int32)
        input_ids_np = np.array(input_ids)
        for row_idx, row in enumerate(input_ids_np):
            positions = np.where(row == assistant_id)[0]
            if positions.size > 0:
                assistant_response_index[row_idx] = positions[0]

        assistant_mask = steps <= mx.array(assistant_response_index).reshape(-1, 1)
        mask = mx.where(assistant_mask, mx.zeros_like(base_mask), base_mask)
    else:
        mask = base_mask

    log_probs = -nn.losses.cross_entropy(logits, targets, reduction="none")
    mask_f = mask.astype(log_probs.dtype)
    token_counts = mx.maximum(mask_f.sum(-1), 1)
    logp_seq_avg = (log_probs * mask_f).sum(-1) / token_counts
    logits_mean = logits.sum() / mx.maximum(mask_f.sum(), 1)
    return logp_seq_avg, logits_mean


orpo_trainer.get_logps = _patched_get_logps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    logger.info(f"{Colors.HEADER}Loading model from {cfg['model']}{Colors.ENDC}")
    model, processor = load(cfg["model"], processor_config={"trust_remote_code": True})
    config = model.config.__dict__

    # No images in this ORPO data; force off so _should_disable_chunked_prefill
    # short-circuits before its .item() call, which MLX disallows inside
    # nn.value_and_grad. Verified via the real traceback, not guessed -- the
    # flag is set once at model init (gemma4_unified.py:68), config-derived,
    # not something forward passes toggle themselves.
    if hasattr(model, "_base_no_chunked_prefill"):
        model._base_no_chunked_prefill = False
        model.no_chunked_prefill = False

    logger.info(f"{Colors.HEADER}Loading local preference data from {cfg['data']}{Colors.ENDC}")
    train_file = cfg.get("train_file", "orpo_stopping_train.jsonl")
    valid_file = cfg.get("valid_file", "orpo_stopping_valid.jsonl")
    train_ds = load_dataset("json", data_files=f"{cfg['data']}/{train_file}", split="train")
    valid_ds = load_dataset("json", data_files=f"{cfg['data']}/{valid_file}", split="train")
    logger.info(f"train: {len(train_ds)} pairs, valid: {len(valid_ds)} pairs")

    train_dataset = PreferenceVisionDataset(train_ds, config, processor)
    val_dataset = PreferenceVisionDataset(valid_ds, config, processor)

    logger.info(f"{Colors.UNDERLINE}Setting up LoRA{Colors.ENDC}")
    lp = cfg["lora_parameters"]
    modules = find_all_linear_names(model.language_model)
    model = get_peft_model(
        model, modules, rank=lp["rank"], alpha=lp["scale"] * lp["rank"],
        dropout=lp.get("dropout", 0.0), verbose=False,
    )
    print_trainable_parameters(model)

    optimizer = optim.Adam(learning_rate=cfg["learning_rate"])

    training_args = ORPOTrainingArgs(
        batch_size=cfg.get("batch_size", 1),
        iters=cfg["iters"],
        steps_per_report=cfg.get("steps_per_report", 5),
        steps_per_eval=cfg.get("steps_per_eval", 50),
        steps_per_save=cfg.get("save_every", 25),
        val_batches=cfg.get("val_batches", 20),
        max_seq_length=cfg.get("max_seq_length", 8192),
        adapter_file=f"{cfg['adapter_path']}/adapters.safetensors",
        grad_checkpoint=cfg.get("grad_checkpoint", True),
        learning_rate=cfg["learning_rate"],
        grad_clip=cfg.get("grad_clip"),
        gradient_accumulation_steps=cfg.get("grad_accumulation_steps", 1),
        full_finetune=False,
        beta=cfg.get("beta", 0.1),
    )

    logger.info(f"{Colors.HEADER}Training model (orpo){Colors.ENDC}")
    train_orpo(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=training_args,
    )
    logger.info(f"{Colors.HEADER}Training completed -> {cfg['adapter_path']}{Colors.ENDC}")


if __name__ == "__main__":
    main()
