"""Launch mlx_vlm LoRA training against local JSONL data.

mlx_vlm.lora's stock CLI only accepts HF Hub datasets (datasets.load_dataset
with a repo id) and hardcodes val_dataset=None -- no validation tracking at
all. This wraps the same underlying pieces (VisionDataset, TrainingArgs,
train()) with local-file loading and real validation wired up.

Needed because mlx_lm can't load this checkpoint at all (model_type
"gemma4_unified" isn't in its supported list -- confirmed by testing, not
assumed), even for a text-only run with no images in the data.

Usage:
    .venv/bin/python scripts/lora_vlm_local.py -c artifacts/lora/lora_gemma4_12b_vlm.yaml
"""
from __future__ import annotations

import argparse
import logging

import mlx.optimizers as optim
import yaml
from datasets import load_dataset

from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    logger.info(f"{Colors.HEADER}Loading model from {cfg['model']}{Colors.ENDC}")
    model, processor = load(cfg["model"], processor_config={"trust_remote_code": True})
    config = model.config.__dict__

    logger.info(f"{Colors.HEADER}Loading local dataset from {cfg['data']}{Colors.ENDC}")
    train_ds = load_dataset("json", data_files=f"{cfg['data']}/train.jsonl", split="train")
    valid_ds = load_dataset("json", data_files=f"{cfg['data']}/valid.jsonl", split="train")
    logger.info(f"train: {len(train_ds)} rows, valid: {len(valid_ds)} rows")

    train_dataset = VisionDataset(
        train_ds, config, processor, train_on_completions=cfg.get("mask_prompt", True)
    )
    val_dataset = VisionDataset(
        valid_ds, config, processor, train_on_completions=cfg.get("mask_prompt", True)
    )

    resume = cfg.get("resume_adapter_file")
    if resume:
        logger.info(f"{Colors.UNDERLINE}Resuming from adapter {resume}{Colors.ENDC}")
        model = apply_lora_layers(model, resume)
    else:
        logger.info(f"{Colors.UNDERLINE}Setting up LoRA{Colors.ENDC}")
        lp = cfg["lora_parameters"]
        modules = find_all_linear_names(model.language_model)
        model = get_peft_model(
            model, modules, rank=lp["rank"], alpha=lp["scale"] * lp["rank"],
            dropout=lp.get("dropout", 0.0), verbose=False,
        )
    print_trainable_parameters(model)

    optimizer = optim.Adam(learning_rate=cfg["learning_rate"])

    training_args = TrainingArgs(
        batch_size=cfg.get("batch_size", 1),
        iters=cfg["iters"],
        steps_per_report=cfg.get("steps_per_report", 10),
        steps_per_eval=cfg.get("steps_per_eval", 200),
        steps_per_save=cfg.get("save_every", 100),
        val_batches=cfg.get("val_batches", 25),
        max_seq_length=cfg.get("max_seq_length", 8192),
        adapter_file=f"{cfg['adapter_path']}/adapters.safetensors",
        grad_checkpoint=cfg.get("grad_checkpoint", True),
        learning_rate=cfg["learning_rate"],
        grad_clip=cfg.get("grad_clip"),
        gradient_accumulation_steps=cfg.get("grad_accumulation_steps", 1),
        full_finetune=False,
    )

    logger.info(f"{Colors.HEADER}Training model (sft){Colors.ENDC}")
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=training_args,
        train_on_completions=cfg.get("mask_prompt", True),
    )
    logger.info(f"{Colors.HEADER}Training completed -> {cfg['adapter_path']}{Colors.ENDC}")


if __name__ == "__main__":
    main()
