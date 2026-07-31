"""QLoRA fine-tune of Gemma-4-31B (dense, QAT checkpoint) via Unsloth.

Runs on the CUDA cloud box, not locally. Trains on the combined greghavens
trace data (Fable-5 + GPT-5.6-Sol + Kimi-K3, ~33.6k rows / ~5,700
trajectories) built by build_lora_data.py --model data/gemma4_tokenizer.

UNVERIFIED AGAINST REAL HARDWARE -- Unsloth's Gemma-4 support has open,
in-progress issues (moe 4-bit expert handling, template/tooling quirks) as
of this writing. This script follows Unsloth's standard, documented QLoRA
pattern, but the whole point of the smoke run below is to catch whatever
doesn't match reality before committing to the full run -- same discipline
as the local MLX smoke tests earlier in this project.

Config choices, and why:
  * rank 8, alpha 16 -- matches greghavens' own fabletron precedent (real,
    published result: BFCL tool-use 31.7->53.6, held-out PPL 3.07->2.18 on
    a similar trace-distillation QLoRA), not a guess.
  * target ALL attention + MLP projections -- this is a DENSE model, no
    routed-expert exclusion logic needed (that was specific to Step-3.7's
    MoE architecture and REAP-pruning saliency findings, irrelevant here).
  * response-only loss via Unsloth's train_on_responses_only, matching the
    Gemma chat template's turn markers.
  * grad_accum 16, matching the fabletron precedent's effective batch size.
  * checkpoint every 25 steps to persistent storage -- non-negotiable if
    running on spot/preemptible (30s eviction notice, ephemeral disk dies
    with the instance).

Usage:
    python cloud_train_gemma4.py --smoke                    # ~30 steps, validate
    python cloud_train_gemma4.py --epochs 1                 # full run
    python cloud_train_gemma4.py --resume --epochs 1        # resume from ckpt
"""
from __future__ import annotations

import argparse
import json
import os

MODEL_PATH = "./gemma4-qat-bf16"  # local path after huggingface-cli download
DATA_DIR = "data/lora_gemma4"     # train.jsonl / valid.jsonl from build_lora_data.py


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default="./adapters/gemma4-agentic")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--smoke", action="store_true",
                    help="cap to ~30 optimizer steps on a small data slice, "
                         "validate the pipeline before the real run")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print(f"[train] loading {a.model_path} (4-bit via Unsloth)...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.model_path,
        max_seq_length=a.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # let Unsloth pick bf16/fp16 for the GPU
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=a.rank,
        lora_alpha=a.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",  # Unsloth's memory-efficient variant
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_train/1e6:.1f}M", flush=True)

    train_rows = load_jsonl(f"{a.data_dir}/train.jsonl")
    valid_rows = load_jsonl(f"{a.data_dir}/valid.jsonl")
    if a.smoke:
        train_rows = train_rows[:200]
        valid_rows = valid_rows[:20]
    print(f"[train] train {len(train_rows)} / valid {len(valid_rows)} rows"
          f"{' (SMOKE subset)' if a.smoke else ''}", flush=True)

    def to_text(row):
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tools=row.get("tools"), tokenize=False)}

    train_ds = Dataset.from_list(train_rows).map(to_text, remove_columns=["messages"])
    valid_ds = Dataset.from_list(valid_rows).map(to_text, remove_columns=["messages"])

    max_steps = 30 if a.smoke else -1

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        dataset_text_field="text",
        max_seq_length=a.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=a.batch_size,
            gradient_accumulation_steps=a.grad_accum,
            num_train_epochs=a.epochs,
            max_steps=max_steps,
            learning_rate=a.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optim="adamw_8bit",
            logging_steps=5 if a.smoke else 10,
            save_steps=a.save_every,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=a.eval_every if not a.smoke else 15,
            output_dir=a.output_dir,
            report_to="none",
            bf16=True,
        ),
    )

    # Response-only loss: mask everything except the model's own turns, so
    # the long agentic context (tool outputs, prior turns) isn't supervised.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    print(f"[train] launching {'SMOKE (30 steps)' if a.smoke else f'{a.epochs} epoch(s)'} ...",
          flush=True)
    trainer.train(resume_from_checkpoint=a.resume)

    print(f"[train] saving adapter -> {a.output_dir}", flush=True)
    model.save_pretrained(a.output_dir)
    tokenizer.save_pretrained(a.output_dir)
    print("[train] done.", flush=True)


if __name__ == "__main__":
    main()
