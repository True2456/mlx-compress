"""DWQ phase 2: train the 4-bit student's affine quant scales to match the
teacher's cached top-k logits (KL). Student is resident (fits M5 RAM); only the
affine scales/biases of quantized layers are trainable.

Usage:
    .venv/bin/python -m reap_stream.dwq_train_student \
        --student models/Step-3.7-p15-4bit \
        --targets artifacts/dwq-targets \
        --out models/Step-3.7-p15-4bit-dwq \
        --lr 1e-5 --epochs 1 --scale 1.0 --grad-checkpoint
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_vlm import load
from mlx_lm.quant.dwq import grad_checkpoint
import mlx_vlm.models.switch_layers as _switch_layers


def _patch_switch_stop_grad():
    """MoE routing indices come from argpartition (non-differentiable). The
    quantized GatherQMM VJP still tries to differentiate rhs_indices and errors
    ('cannot compute the gradient wrt the indices'). Stop-gradient the indices
    at the SwitchGLU boundary so backprop reaches only the quant scales."""
    orig = _switch_layers.SwitchGLU.__call__
    if getattr(orig, "_dwq_patched", False):
        return
    def call(self, x, indices, *args, **kwargs):
        return orig(self, x, mx.stop_gradient(indices), *args, **kwargs)
    call._dwq_patched = True
    _switch_layers.SwitchGLU.__call__ = call


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _unfreeze_affine_scales(model, keys=("scales", "biases")):
    """Freeze everything, then unfreeze the given quant params on affine <8-bit
    modules. Use ("scales",) to halve the trainable/optimizer footprint."""
    model.freeze()
    keys = list(keys)
    def visit(_, m):
        if (hasattr(m, "bits") and hasattr(m, "group_size")
                and getattr(m, "mode", "affine") == "affine" and m.bits < 8):
            m.unfreeze(keys=keys, recurse=False)
            return 1
        return 0
    # apply_to_modules walks the tree; count via closure
    counts = []
    def walk(path, module):
        counts.append(visit(path, module))
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
        files = [f for f in files
                 if int(Path(f).stem) not in excluded]
        print(f"[dwq-train] excluded {before - len(files)} prompts via "
              f"{exclude_path.name} (e.g. misaligned multimodal rows)", flush=True)
    return files


def _kl_topk(student_logits, tgt_idx, tgt_vals, mask, scale):
    """KL(teacher || student) over the teacher's top-k indices, renormalized,
    averaged over real (unpadded) positions only.

    student_logits: [seq, vocab]; tgt_idx/tgt_vals: [seq, k]; mask: [seq].
    """
    s = mx.take_along_axis(student_logits, tgt_idx, axis=-1)   # [seq, k]
    s = s * scale
    t = tgt_vals.astype(mx.float32) * scale
    log_p_t = t - mx.logsumexp(t, axis=-1, keepdims=True)
    log_p_s = s - mx.logsumexp(s, axis=-1, keepdims=True)
    p_t = mx.exp(log_p_t)
    per_pos = (p_t * (log_p_t - log_p_s)).sum(axis=-1)         # [seq]
    return (per_pos * mask).sum() / mx.maximum(mask.sum(), 1)


def _ckpt_paths(ckpt_dir: Path):
    return ckpt_dir / "trainable_scales.safetensors", ckpt_dir / "state.json"


def _save_checkpoint(ckpt_dir: Path, lm, step: int, meta: dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path, state_path = _ckpt_paths(ckpt_dir)
    tmp_w = weights_path.with_name(weights_path.stem + ".tmp.safetensors")
    flat = dict(tree_flatten(lm.trainable_parameters()))
    mx.save_safetensors(str(tmp_w), flat)
    tmp_w.replace(weights_path)
    state_path.write_text(json.dumps({"step": step, **meta}, indent=2))


def _load_checkpoint(ckpt_dir: Path, lm):
    weights_path, state_path = _ckpt_paths(ckpt_dir)
    if not (weights_path.exists() and state_path.exists()):
        return 0
    state = json.loads(state_path.read_text())
    saved = mx.load(str(weights_path))
    lm.update(tree_unflatten(list(saved.items())))
    mx.eval(lm.parameters())
    print(f"[dwq-train] resumed from checkpoint at step {state['step']}", flush=True)
    return state["step"]


def train(student_path, targets_dir, out_dir, lr, epochs, scale, use_ckpt, max_steps=0, scales_only=False, optimizer="adam", max_prompts=0, ckpt_dir=None, ckpt_every=25, grad_accum=1):
    _patch_switch_stop_grad()
    print(f"[dwq-train] loading student (resident): {student_path}", flush=True)
    model, processor = load(student_path, lazy=False)
    text = _text_model(model)
    lm = model.language_model

    keys = ("scales",) if scales_only else ("scales", "biases")
    n_trainable = _unfreeze_affine_scales(lm, keys)
    n_params = sum(v.size for _, v in tree_flatten(lm.trainable_parameters()))
    print(f"[dwq-train] unfroze {n_trainable} affine modules, "
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
    k = int(meta.get("topk", 128))
    print(f"[dwq-train] {len(files)} target prompts, lr={lr}, epochs={epochs}, "
          f"padding all to seq={pad_len} (one compile)", flush=True)

    ck_dir = Path(ckpt_dir) if ckpt_dir else Path(out_dir + "-ckpt")
    resume_step = _load_checkpoint(ck_dir, lm)

    def _pad(d):
        """Right-pad input_ids / topk to pad_len; return arrays + float mask."""
        n = d["input_ids"].shape[0]
        pad = max(pad_len - n, 0)
        inp = mx.pad(d["input_ids"], (0, pad))[:pad_len][None]
        idx = mx.pad(d["topk_idx"], [(0, pad), (0, 0)])[:pad_len]
        vals = mx.pad(d["topk_vals"], [(0, pad), (0, 0)])[:pad_len]
        mask = mx.concatenate([mx.ones(min(n, pad_len)),
                               mx.zeros(max(pad_len - n, 0))])[:pad_len]
        return inp, idx, vals, mask

    opt = optim.SGD(learning_rate=lr) if optimizer == "sgd" else optim.Adam(learning_rate=lr)
    print(f"[dwq-train] optimizer={optimizer}", flush=True)

    def loss_fn(inp, idx, vals, mask):
        out = lm(input_ids=inp)
        return _kl_topk(out.logits[0], idx, vals, mask, scale)

    lg = nn.value_and_grad(lm, loss_fn)

    t0 = time.time()
    step = 0            # counts prompts consumed (drives resume/checkpoint cadence)
    opt_steps = 0        # counts actual optimizer updates (every grad_accum prompts)
    log_every = 10
    ckpt_meta = {"student": student_path, "targets": targets_dir,
                 "n_prompts": len(files), "scales_only": scales_only,
                 "optimizer": optimizer, "lr": lr, "grad_accum": grad_accum}
    stop = False
    accum_grads = None
    accum_loss = 0.0
    accum_n = 0
    ts = time.time()
    for ep in range(epochs):
        for f in files:
            step += 1
            if step <= resume_step:
                continue   # already trained on this prompt before interruption
            inp, idx, vals, mask = _pad(mx.load(f))
            loss, grads = lg(inp, idx, vals, mask)

            if grad_accum <= 1:
                # fast path: no accumulator, no extra fp16/gc/clear_cache tax.
                # Measured: holding even one extra grad-tree copy (fp32 or
                # fp16) pushes active memory 109->115-130GB, right at this
                # box's ~115-118GB wired ceiling, causing 3x+ slowdown from
                # memory pressure. Only pay that cost when actually accumulating.
                opt.update(lm, grads)
                mx.eval(lm.parameters(), opt.state, loss)
                opt_steps += 1
                avg_loss = loss.item()
                if opt_steps <= 3 or opt_steps % log_every == 0:
                    act = mx.get_active_memory() / 1e9
                    print(f"[dwq-train] ep{ep} prompt {step}/{len(files)} "
                          f"kl={avg_loss:.4f} ({time.time()-ts:.1f}s, active={act:.0f} GB)",
                          flush=True)
                ts = time.time()
            else:
                mx.eval(loss, grads)
                flat_g = {k: v.astype(mx.float16) for k, v in tree_flatten(grads)}
                if accum_grads is None:
                    new_accum = flat_g
                else:
                    new_accum = {k: accum_grads[k] + flat_g[k] for k in flat_g}
                mx.eval(list(new_accum.values()))
                accum_grads = new_accum
                del new_accum, flat_g, grads
                gc.collect()
                mx.clear_cache()
                accum_loss += loss.item()
                accum_n += 1

                is_last = (ep == epochs - 1) and (f == files[-1])
                if accum_n >= grad_accum or is_last:
                    avg_grads = tree_unflatten(
                        [(k, (v / accum_n).astype(mx.float32))
                         for k, v in accum_grads.items()])
                    opt.update(lm, avg_grads)
                    mx.eval(lm.parameters(), opt.state)
                    opt_steps += 1
                    avg_loss = accum_loss / accum_n
                    if opt_steps <= 3 or opt_steps % (log_every) == 0:
                        act = mx.get_active_memory() / 1e9
                        print(f"[dwq-train] ep{ep} prompt {step}/{len(files)} "
                              f"opt_step {opt_steps} avg_kl={avg_loss:.4f} "
                              f"(n={accum_n}, {time.time()-ts:.1f}s, active={act:.0f} GB)",
                              flush=True)
                    accum_grads = None
                    accum_loss = 0.0
                    accum_n = 0
                    ts = time.time()
            if step % ckpt_every == 0:
                _save_checkpoint(ck_dir, lm, step, ckpt_meta)
                print(f"[dwq-train] checkpoint saved at step {step} -> {ck_dir}", flush=True)
            if max_steps and step >= max_steps:
                print(f"[dwq-train] hit max_steps={max_steps}, stopping", flush=True)
                stop = True
                break
        if stop:
            break
    _save_checkpoint(ck_dir, lm, step, ckpt_meta)
    print(f"[dwq-train] final checkpoint saved at step {step} "
          f"({opt_steps} optimizer updates) -> {ck_dir}", flush=True)

    # save: re-quantized student with trained scales
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[dwq-train] saving -> {out}", flush=True)
    from mlx_vlm.utils import save_weights, save_config
    target = model
    save_weights(out, target, donate_weights=False)
    for pat in ("*.py", "*.json"):
        for fp in glob.glob(str(Path(student_path) / pat)):
            if Path(fp).name == "model.safetensors.index.json":
                continue
            shutil.copy(fp, out)
    for item in Path(student_path).iterdir():
        if item.is_dir():
            dst = out / item.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
    processor.save_pretrained(out)
    cfg = json.loads((Path(student_path) / "config.json").read_text())
    save_config(cfg, out / "config.json")
    print(f"[dwq-train] done in {round(time.time()-t0,1)}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--scales-only", action="store_true")
    ap.add_argument("--optimizer", choices=["adam","sgd"], default="adam")
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=None,
                    help="Defaults to <out>-ckpt. Stores only the small trainable "
                         "scale tensors + step count, for cheap periodic saves.")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--grad-accum", type=int, default=1)
    a = ap.parse_args()
    train(a.student, a.targets, a.out, a.lr, a.epochs, a.scale, a.grad_checkpoint,
          a.max_steps, a.scales_only, a.optimizer, a.max_prompts,
          a.ckpt_dir, a.ckpt_every, a.grad_accum)
