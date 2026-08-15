"""DWQ phase 2 for Ling-3.0-flash: train the quantized student's affine quant
scales to match the teacher's cached top-k logits (KL). Student is resident
(the quantized checkpoint is small enough to fit, unlike the 237GB BF16
teacher); only the affine scales/biases of quantized layers are trainable.

Port of dwq_train_student.py onto bailing_hybrid / mlx_lm (the step3p7
version loads via mlx_vlm and expects a `.logits` attribute; bailing_hybrid's
Model.__call__ returns the logits array directly).

Usage:
    .venv/bin/python -m reap_stream.dwq_train_student_ling3 \
        --student models/Ling-3.0-p_-4bit \
        --targets artifacts/dwq-targets-ling3 \
        --out models/Ling-3.0-p_-4bit-dwq \
        --lr 1e-6 --epochs 1 --scale 1.0 --scales-only --optimizer sgd
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.utils import load, save
import mlx_lm.models.switch_layers as _switch_layers

from .bailing_swiglu_clamp import apply_bailing_swiglu_clamp


def _patch_switch_stop_grad():
    """MoE routing indices come from argpartition (non-differentiable) in
    bailing_hybrid's group_expert_select too; same GatherQMM::vjp issue as
    step3p7. Stop-gradient the indices at the SwitchGLU boundary so backprop
    reaches only the quant scales."""
    orig = _switch_layers.SwitchGLU.__call__
    if getattr(orig, "_dwq_patched", False):
        return
    def call(self, x, indices, *args, **kwargs):
        return orig(self, x, mx.stop_gradient(indices), *args, **kwargs)
    call._dwq_patched = True
    _switch_layers.SwitchGLU.__call__ = call


def _unfreeze_affine_scales(model, keys=("scales", "biases")):
    """Freeze everything, then unfreeze the given quant params on affine <8-bit
    modules. Use ("scales",) to halve the trainable/optimizer footprint."""
    model.freeze()
    keys = list(keys)
    counts = []
    def walk(path, m):
        if (hasattr(m, "bits") and hasattr(m, "group_size")
                and getattr(m, "mode", "affine") == "affine" and m.bits < 8):
            m.unfreeze(keys=keys, recurse=False)
            counts.append(1)
    model.apply_to_modules(walk)
    return sum(counts)


def _load_targets(targets_dir):
    files = sorted(glob.glob(str(Path(targets_dir) / "[0-9]*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no target files in {targets_dir}")
    exclude_path = Path(targets_dir) / "exclude_indices.json"
    if exclude_path.exists():
        excluded = set(json.loads(exclude_path.read_text()))
        before = len(files)
        files = [f for f in files if int(Path(f).stem) not in excluded]
        print(f"[dwq-train-ling3] excluded {before - len(files)} prompts via "
              f"{exclude_path.name}", flush=True)
    return files


def _kl_topk(student_logits, tgt_idx, tgt_vals, mask, scale):
    """KL(teacher || student) over the teacher's top-k indices, renormalized,
    averaged over real (unpadded) positions only.

    student_logits: [seq, vocab]; tgt_idx/tgt_vals: [seq, k]; mask: [seq].
    """
    s = mx.take_along_axis(student_logits, tgt_idx, axis=-1) * scale
    t = tgt_vals.astype(mx.float32) * scale
    log_p_t = t - mx.logsumexp(t, axis=-1, keepdims=True)
    log_p_s = s - mx.logsumexp(s, axis=-1, keepdims=True)
    p_t = mx.exp(log_p_t)
    per_pos = (p_t * (log_p_t - log_p_s)).sum(axis=-1)
    return (per_pos * mask).sum() / mx.maximum(mask.sum(), 1)


def _ckpt_paths(ckpt_dir: Path):
    return ckpt_dir / "trainable_scales.safetensors", ckpt_dir / "state.json"


def _save_checkpoint(ckpt_dir: Path, model, step: int, meta: dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path, state_path = _ckpt_paths(ckpt_dir)
    tmp_w = weights_path.with_name(weights_path.stem + ".tmp.safetensors")
    flat = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(tmp_w), flat)
    tmp_w.replace(weights_path)
    state_path.write_text(json.dumps({"step": step, **meta}, indent=2))


def _load_checkpoint(ckpt_dir: Path, model):
    weights_path, state_path = _ckpt_paths(ckpt_dir)
    if not (weights_path.exists() and state_path.exists()):
        return 0
    state = json.loads(state_path.read_text())
    saved = mx.load(str(weights_path))
    model.update(tree_unflatten(list(saved.items())))
    mx.eval(model.parameters())
    print(f"[dwq-train-ling3] resumed from checkpoint at step {state['step']}", flush=True)
    return state["step"]


def train(student_path, targets_dir, out_dir, lr, epochs, scale, use_ckpt,
          max_steps=0, scales_only=False, optimizer="adam", max_prompts=0,
          ckpt_dir=None, ckpt_every=25):
    apply_bailing_swiglu_clamp()
    _patch_switch_stop_grad()
    print(f"[dwq-train-ling3] loading student (resident): {student_path}", flush=True)
    model, tokenizer = load(student_path, lazy=False)
    text = model.model

    # bailing_hybrid's KimiDeltaAttention picks its gated-delta backend via
    # `use_kernel = not self.training`: the fast Metal kernel path has no
    # backward pass (Primitive::vjp not implemented for CustomKernel), only
    # the plain-MLX-ops path (gated_delta_ops) is differentiable. Must be in
    # train mode before any forward we intend to backprop through.
    model.train()

    keys = ("scales",) if scales_only else ("scales", "biases")
    n_trainable = _unfreeze_affine_scales(model, keys)
    n_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"[dwq-train-ling3] unfroze {n_trainable} affine modules, "
          f"{n_params/1e6:.1f}M trainable scale params", flush=True)

    if use_ckpt:
        for layer in text.layers:
            if layer is not None:
                grad_checkpoint(layer)

    files = _load_targets(targets_dir)
    if max_prompts:
        files = files[:max_prompts]
    meta = json.loads((Path(targets_dir) / "targets_meta.json").read_text())
    pad_len = int(meta.get("max_tokens", 384))
    print(f"[dwq-train-ling3] {len(files)} target prompts, lr={lr}, epochs={epochs}, "
          f"padding all to seq={pad_len} (one compile)", flush=True)

    ck_dir = Path(ckpt_dir) if ckpt_dir else Path(out_dir + "-ckpt")
    resume_step = _load_checkpoint(ck_dir, model)

    def _pad(d):
        n = d["input_ids"].shape[0]
        pad = max(pad_len - n, 0)
        inp = mx.pad(d["input_ids"], (0, pad))[:pad_len][None]
        idx = mx.pad(d["topk_idx"], [(0, pad), (0, 0)])[:pad_len]
        vals = mx.pad(d["topk_vals"], [(0, pad), (0, 0)])[:pad_len]
        mask = mx.concatenate([mx.ones(min(n, pad_len)),
                               mx.zeros(max(pad_len - n, 0))])[:pad_len]
        return inp, idx, vals, mask

    opt = optim.SGD(learning_rate=lr) if optimizer == "sgd" else optim.Adam(learning_rate=lr)
    print(f"[dwq-train-ling3] optimizer={optimizer}", flush=True)

    def loss_fn(inp, idx, vals, mask):
        logits = model(inp)          # bailing_hybrid Model.__call__ -> logits directly
        return _kl_topk(logits[0], idx, vals, mask, scale)

    lg = nn.value_and_grad(model, loss_fn)

    t0 = time.time()
    step = 0
    opt_steps = 0
    log_every = 10
    ckpt_meta = {"student": student_path, "targets": targets_dir,
                 "n_prompts": len(files), "scales_only": scales_only,
                 "optimizer": optimizer, "lr": lr}
    stop = False
    ts = time.time()
    for ep in range(epochs):
        for f in files:
            step += 1
            if step <= resume_step:
                continue
            inp, idx, vals, mask = _pad(mx.load(f))
            loss, grads = lg(inp, idx, vals, mask)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            opt_steps += 1
            if opt_steps <= 3 or opt_steps % log_every == 0:
                act = mx.get_active_memory() / 1e9
                print(f"[dwq-train-ling3] ep{ep} prompt {step}/{len(files)} "
                      f"kl={loss.item():.4f} ({time.time()-ts:.1f}s, active={act:.0f} GB)",
                      flush=True)
            ts = time.time()
            if step % ckpt_every == 0:
                _save_checkpoint(ck_dir, model, step, ckpt_meta)
                print(f"[dwq-train-ling3] checkpoint saved at step {step} -> {ck_dir}", flush=True)
            if max_steps and step >= max_steps:
                print(f"[dwq-train-ling3] hit max_steps={max_steps}, stopping", flush=True)
                stop = True
                break
        if stop:
            break
    _save_checkpoint(ck_dir, model, step, ckpt_meta)
    print(f"[dwq-train-ling3] final checkpoint saved at step {step} "
          f"({opt_steps} optimizer updates) -> {ck_dir}", flush=True)

    out = Path(out_dir)
    print(f"[dwq-train-ling3] saving -> {out}", flush=True)
    cfg = json.loads((Path(student_path) / "config.json").read_text())
    save(out, student_path, model, tokenizer, cfg)
    print(f"[dwq-train-ling3] done in {round(time.time()-t0,1)}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--scales-only", action="store_true")
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-every", type=int, default=25)
    a = ap.parse_args()
    train(a.student, a.targets, a.out, a.lr, a.epochs, a.scale, a.grad_checkpoint,
          a.max_steps, a.scales_only, a.optimizer, a.max_prompts,
          a.ckpt_dir, a.ckpt_every)
