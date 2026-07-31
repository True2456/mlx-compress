"""Text-only LoRA SFT for Step-3.7 (mlx_vlm-native).

mlx_lm.lora can't load step3p7 (it's a VLM; mlx_lm has no step3p7), and
mlx_vlm.lora's data pipeline is image-text oriented. So this is a small
custom trainer that reuses the proven local pattern:

  * load via mlx_vlm.load (handles step3p7), forward through
    model.language_model (text-only, no pixel_values) -- same as the DWQ
    trainer;
  * attach + save + reload LoRA entirely via mlx_vlm.trainer utilities
    (_apply_lora_layers / save_adapter / apply_lora_layers) so the adapter
    round-trips cleanly for `mlx_vlm.load(adapter_path=...)` and the
    --adapter-path evals;
  * DWQ-proven memory discipline: pad every batch to a fixed length (one
    compile), plain optimizer (no grad-accum tax), mx.clear_cache in the hot
    loop, footprint-based memory logging;
  * mask_prompt: loss on the final assistant turn only, located by the
    prompt-prefix method (render messages[:-1] with add_generation_prompt,
    completion = the trailing tokens).

Usage:
    .venv/bin/python -m reap_stream.lora_sft_step37 \
        --model models/Step-3.7-p15-4bit-vblend-shared8 \
        --data data/lora_step37 --adapter-dir artifacts/lora/adapters \
        --iters 30 --lr 2e-5 --max-seq-length 8192
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm import load
import mlx_vlm.models.switch_layers as _switch_layers
from mlx_vlm.trainer.utils import _apply_lora_layers, save_adapter
from mlx_lm.quant.dwq import grad_checkpoint


def _patch_switch_stop_grad():
    """MoE routing indices come from argpartition (non-differentiable). The
    quantized GatherQMM VJP still tries to differentiate rhs_indices and errors
    ('cannot compute the gradient wrt the indices'). Stop-gradient the indices
    at the SwitchGLU boundary so backprop reaches only the LoRA params. (Same
    fix DWQ needed on this model -- FINDINGS.md sec 7.)"""
    orig = _switch_layers.SwitchGLU.__call__
    if getattr(orig, "_sg_patched", False):
        return
    def call(self, x, indices, *args, **kwargs):
        return orig(self, x, mx.stop_gradient(indices), *args, **kwargs)
    call._sg_patched = True
    _switch_layers.SwitchGLU.__call__ = call

SHORT_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "self_attn.g_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "mlp.share_expert.gate_proj", "mlp.share_expert.up_proj",
    "mlp.share_expert.down_proj", "mlp.gate.gate",
]


def full_lora_keys(model):
    """Full dotted paths for every module whose path ends in a target key,
    excluding the routed experts (mlp.switch_mlp.*)."""
    keys = []
    for name, _ in model.named_modules():
        if ".switch_mlp." in name:
            continue
        if any(name.endswith(sk) for sk in SHORT_KEYS):
            keys.append(name)
    return keys


def build_lora_config(model, rank, scale, dropout):
    cfg = {
        "fine_tune_type": "lora",
        "num_layers": -1,
        "lora_parameters": {
            "rank": rank, "scale": scale, "dropout": dropout,
            "keys": full_lora_keys(model),
        },
    }
    return cfg


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_row(tok, row, max_len):
    """Return (ids, mask) where mask=1 on completion (final assistant) tokens.
    None if the row can't be located/rendered or has no completion in-window."""
    msgs = row["messages"]
    tools = row.get("tools")
    try:
        full = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:-1], tools=tools,
                                         add_generation_prompt=True, tokenize=False)
    except Exception:
        return None
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    # completion = trailing tokens after the (verified) prompt prefix
    p = len(prompt_ids)
    if p >= len(full_ids):
        return None
    if full_ids[:p] != prompt_ids:
        # fall back to longest common prefix if the render isn't a clean prefix
        p = 0
        for a, b in zip(full_ids, prompt_ids):
            if a != b:
                break
            p += 1
        if p >= len(full_ids):
            return None
    # TAIL truncation, not head: the completion (final assistant turn) is at
    # the END. Head-truncating a long row would drop the very tokens we
    # supervise. Keep the last max_len tokens so the completion is always
    # retained, with as much preceding context as fits. (Same head/tail lesson
    # as the PPL work -- the right end to keep is task-dependent.)
    n_full = len(full_ids)
    if n_full > max_len:
        cut = n_full - max_len
        ids = full_ids[cut:]
        p = max(p - cut, 0)   # shift completion boundary into the kept window
    else:
        ids = full_ids
    mask = [0] * min(p, len(ids)) + [1] * max(0, len(ids) - p)
    if sum(mask) == 0:
        return None  # completion longer than the whole window (shouldn't happen)
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dir with train.jsonl / valid.jsonl")
    ap.add_argument("--adapter-dir", default="artifacts/lora/adapters")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--val-batches", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--max-rows", type=int, default=0, help="cap train rows (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mx.random.seed(a.seed)
    np.random.seed(a.seed)

    _patch_switch_stop_grad()
    print(f"[lora] loading {a.model} (resident)", flush=True)
    model, processor = load(a.model, lazy=False)
    tok = getattr(processor, "tokenizer", processor)

    cfg = build_lora_config(model, a.rank, a.scale, a.dropout)
    n_keys = len(cfg["lora_parameters"]["keys"])
    print(f"[lora] attaching LoRA to {n_keys} modules "
          f"(rank={a.rank}, scale={a.scale})", flush=True)
    _apply_lora_layers(model, cfg)
    model.config.lora = cfg  # so save_adapter writes a reload-compatible config

    # Freeze EVERYTHING, then unfreeze only the lora_a/lora_b LEAF params.
    # The LoRALinear wraps the frozen 4-bit base as `linear.{weight,scales,
    # biases}` plus trainable `lora_a`/`lora_b`; a recursive unfreeze on the
    # whole module would re-enable the base weights and push gradient into the
    # quantized matmul (QuantizedMatmul::vjp crash). Partial, non-recursive
    # unfreeze by key keeps the base frozen.
    model.freeze()
    def _unfreeze_lora(_, m):
        if type(m).__name__ == "LoRALinear" and hasattr(m, "lora_a"):
            m.unfreeze(recurse=False, keys=["lora_a", "lora_b"])
    model.apply_to_modules(_unfreeze_lora)
    lm = model.language_model

    # Gradient checkpointing per decoder layer -- ESSENTIAL here. Backprop to
    # the earliest LoRA layer needs activations for the whole 45-layer stack;
    # without recompute, that graph hit 241GB (>128GB RAM) at seq 8192 and
    # thrashed to death. Recompute-on-backward trades compute for memory.
    # (Same mechanism DWQ used on this model.)
    _text = getattr(lm, "model", lm)
    n_ckpt = 0
    for layer in _text.layers:
        if layer is not None:
            grad_checkpoint(layer)
            n_ckpt += 1
    print(f"[lora] gradient checkpointing on {n_ckpt} layers", flush=True)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"[lora] trainable params: {n_train/1e6:.2f}M", flush=True)

    train_rows = load_rows(Path(a.data) / "train.jsonl")
    valid_rows = load_rows(Path(a.data) / "valid.jsonl")
    if a.max_rows:
        train_rows = train_rows[:a.max_rows]
    print(f"[lora] train {len(train_rows)} / valid {len(valid_rows)} rows; "
          f"pad to seq={a.max_seq_length}", flush=True)

    pad_len = a.max_seq_length

    def batch_from(row):
        tk = tokenize_row(tok, row, pad_len)
        if tk is None:
            return None
        ids, mask = tk
        n = len(ids)
        pad = max(pad_len - n, 0)
        inp = mx.array(ids + [0] * pad)[None]
        m = mx.array(mask + [0.0] * pad, dtype=mx.float32)
        return inp, m, n

    def loss_fn(inp, m):
        logits = lm(input_ids=inp).logits[0].astype(mx.float32)
        # logits[t] predicts token t+1; supervise where target (t+1) is completion
        tgt = inp[0, 1:]
        lg = logits[:-1]
        mtar = m[1:]
        lse = mx.logsumexp(lg, axis=-1)
        picked = mx.take_along_axis(lg, tgt[:, None], axis=-1)[:, 0]
        nll = (lse - picked) * mtar
        return nll.sum() / mx.maximum(mtar.sum(), 1.0)

    lg_fn = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=a.lr)

    ck = Path(a.adapter_dir)
    ck.mkdir(parents=True, exist_ok=True)

    def run_eval():
        idx = np.random.RandomState(0).permutation(len(valid_rows))[:a.val_batches]
        tot, k = 0.0, 0
        for j in idx:
            b = batch_from(valid_rows[int(j)])
            if b is None:
                continue
            inp, m, _ = b
            tot += float(loss_fn(inp, m).item())
            k += 1
            mx.clear_cache()
        return tot / max(k, 1)

    order = np.random.permutation(len(train_rows))
    t0 = time.time()
    ptr = 0
    ts = time.time()
    for it in range(1, a.iters + 1):
        b = None
        while b is None:  # skip unrenderable/empty-completion rows
            if ptr >= len(order):
                order = np.random.permutation(len(train_rows))
                ptr = 0
            b = batch_from(train_rows[int(order[ptr])])
            ptr += 1
        inp, m, n = b
        loss, grads = lg_fn(inp, m)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state, loss)
        mx.clear_cache()

        if it <= 3 or it % a.steps_per_report == 0:
            act = mx.get_active_memory() / 1e9
            print(f"[lora] iter {it}/{a.iters} loss={loss.item():.4f} "
                  f"seq={n} ({time.time()-ts:.1f}s, active={act:.0f}GB)", flush=True)
            ts = time.time()

        if a.steps_per_eval and it % a.steps_per_eval == 0:
            vl = run_eval()
            print(f"[lora] iter {it} VAL loss={vl:.4f}", flush=True)

        if a.save_every and it % a.save_every == 0:
            save_adapter(model, ck / "adapters.safetensors")
            print(f"[lora] saved adapter at iter {it} -> {ck}", flush=True)

    save_adapter(model, ck / "adapters.safetensors")
    print(f"[lora] done in {time.time()-t0:.0f}s, final adapter -> {ck}", flush=True)


if __name__ == "__main__":
    main()
