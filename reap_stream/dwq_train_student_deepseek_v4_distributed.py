"""Distributed (2-machine) DWQ phase 2 for DeepSeek-V4-Flash: same KL-to-teacher
training as dwq_train_student_deepseek_v4.py, but with the model's layers
split across two machines over Thunderbolt so the student's memory footprint
divides between them instead of requiring the full model resident on one box.

Why this needed new work beyond reusing the AWQ distributed script
(awq_quantize_deepseek_v4_distributed.py): that script only sends activations
FORWARD across the machine boundary (calibration, no backward pass). DWQ
needs gradients to flow back through the same boundary. Checked directly:
mx.distributed.send/recv_like have no registered VJP rule (confirmed via
test_dwq_builtin_pipeline_grad.py -- "[Primitive::vjp] Not implemented for
Send"), so DeepseekV4Model.__call__'s own built-in pipeline path (which uses
those primitives internally) can't just be wrapped in nn.value_and_grad.

The actual mechanism, validated against a same-process reference in
test_dwq_distributed_backward.py before use here: treat the boundary hidden
state as a plain (non-parameter) function argument. Rank 1 (early layers)
computes it, sends it as a raw un-tracked tensor, then LATER receives the
upstream gradient dh and calls mx.vjp on its own forward function with dh as
the seed cotangent to get its param gradients. Rank 0 (late layers + head)
receives h as an ordinary input, computes loss, and calls mx.vjp treating
(h, its own trainable params) as the differentiated primals in one call --
getting both dh (to send back) and its own gradients simultaneously. This is
standard reverse-mode chain rule at a pipeline boundary; the toy test matched
a single-process mx.value_and_grad reference to within float32 rounding
(3-4 significant figures, not a real mismatch -- see that test's comments).

Reversed pipeline convention matches PipelineMixin/AWQ's distributed script:
rank 0 owns the LAST layers (+ hc_head/norm/lm_head, so only rank 0 can
compute the actual loss value), rank 1 owns the FIRST layers.

Each rank checkpoints only its own local trainable scales/biases -- see
merge_distributed_dwq_checkpoints() (bottom of this file) to assemble both
ranks' shards into one final loadable student checkpoint.

Usage:
    mlx.launch --hosts <rank0-ip>,<rank1-ip> --backend ring -- \
        .venv/bin/python -m reap_stream.dwq_train_student_deepseek_v4_distributed \
        --student models/DeepSeek-V4-Flash-0731-awq2bit3bit-v2 \
        --targets artifacts/dwq-targets-v2-agentic \
        --ckpt-dir /tmp/dwq_dist_ckpt \
        --layer-split 29,14 --lr 1e-5 --max-steps 100
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
import mlx_vlm.models.switch_layers as _switch_layers


def _patch_switch_stop_grad():
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
    model.freeze()
    keys = list(keys)
    counts = []
    def visit(_, m):
        if (hasattr(m, "bits") and hasattr(m, "group_size")
                and getattr(m, "mode", "affine") == "affine" and m.bits < 8):
            m.unfreeze(keys=keys, recurse=False)
            counts.append(1)
    model.apply_to_modules(visit)
    return len(counts)


def _kl_topk(student_logits, tgt_idx, tgt_vals, mask, scale, tgt_logz=None,
             special_ids=(1,), special_weight=1.0):
    """KL to the teacher over its top-k tokens.

    When `tgt_logz` (the teacher's full-vocab logsumexp) is supplied, the loss
    becomes a (k+1)-way KL: the top-k tokens plus one "everything else" bucket
    holding the residual mass. This matters. Without that bucket the objective
    never observes the ~129k tokens outside the top-k, so the student can
    inflate them for free -- measured after 100 steps: top-k KL improved 42%
    while out-of-top-k mass rose 84% and P(EOS) rose 63x, which made
    <|end_of_sentence|> the argmax at agentic continuation points and stopped
    generation dead (7/8 probes emitted nothing). Falls back to the old
    renormalized top-k KL when logz is unavailable, so pre-existing target sets
    still load.
    """
    s = mx.take_along_axis(student_logits, tgt_idx, axis=-1) * scale
    t = tgt_vals.astype(mx.float32) * scale

    if tgt_logz is None:
        log_p_t = t - mx.logsumexp(t, axis=-1, keepdims=True)
        log_p_s = s - mx.logsumexp(s, axis=-1, keepdims=True)
        p_t = mx.exp(log_p_t)
        per_pos = (p_t * (log_p_t - log_p_s)).sum(axis=-1)
        return (per_pos * mask).sum() / mx.maximum(mask.sum(), 1)

    # Teacher: true probabilities against its own full-vocab partition, plus
    # the leftover mass as a single extra category.
    tz = (tgt_logz.astype(mx.float32) * scale)[..., None]
    log_p_t = t - tz
    p_t = mx.exp(log_p_t)
    rest_t = mx.maximum(1.0 - p_t.sum(axis=-1), 1e-6)

    # Student: same construction against its own full-vocab partition.
    sz = mx.logsumexp(student_logits.astype(mx.float32) * scale, axis=-1, keepdims=True)
    log_p_s = s - sz
    p_s = mx.exp(log_p_s)
    rest_s = mx.maximum(1.0 - p_s.sum(axis=-1), 1e-6)

    per_pos = (p_t * (log_p_t - log_p_s)).sum(axis=-1)
    per_pos = per_pos + rest_t * (mx.log(rest_t) - mx.log(rest_s))

    # One-sided bound on special tokens (EOS). The rest bucket above constrains
    # only the AGGREGATE mass outside the top-k, and its gradient is scaled by
    # the teacher's own rest mass (~6e-4) -- ~1000x too weak to stop the student
    # concentrating that budget on a single token. Measured: EOS reached p=0.33
    # (rank 1) at agentic continuation points while the teacher never exceeds
    # 0.011 there (top-1 EOS in 0/62 sampled positions).
    #
    # When a special token is ABSENT from the teacher's top-k, its true teacher
    # probability is provably below the smallest top-k probability. That gives a
    # free upper bound needing no extra collected data, and unlike the KL terms
    # its gradient does not vanish as the teacher's mass goes to zero.
    if special_ids:
        p_min = mx.exp(log_p_t[..., -1])
        for tid in special_ids:
            absent = 1.0 - mx.any(tgt_idx == tid, axis=-1).astype(mx.float32)
            p_s_tid = mx.exp(student_logits[..., tid].astype(mx.float32) * scale - sz[..., 0])
            per_pos = per_pos + special_weight * absent * mx.maximum(p_s_tid - p_min, 0.0)
    return (per_pos * mask).sum() / mx.maximum(mask.sum(), 1)


def _load_targets(targets_dir):
    files = sorted(glob.glob(str(Path(targets_dir) / "[0-9]*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no target files in {targets_dir}")
    return files


def _pad(d, pad_len):
    n = d["input_ids"].shape[0]
    pad = max(pad_len - n, 0)
    inp = mx.pad(d["input_ids"], (0, pad))[:pad_len][None]
    idx = mx.pad(d["topk_idx"], [(0, pad), (0, 0)])[:pad_len]
    vals = mx.pad(d["topk_vals"], [(0, pad), (0, 0)])[:pad_len]
    mask = mx.concatenate([mx.ones(min(n, pad_len)), mx.zeros(max(pad_len - n, 0))])[:pad_len]
    logz = d.get("logz")
    if logz is not None:
        logz = mx.pad(logz, (0, pad))[:pad_len]
    return inp, idx, vals, mask, logz


def _ckpt_paths(ckpt_dir: Path):
    return (ckpt_dir / "trainable_scales.safetensors",
            ckpt_dir / "state.json",
            ckpt_dir / "optimizer.safetensors")


def _save_checkpoint(ckpt_dir: Path, lm, opt, step: int, meta: dict):
    """Params AND optimizer state. The Thunderbolt ring link drops
    occasionally (observed: EPIPE mid-run, 'Too many send/recv errors'), so
    runs get restarted; without the optimizer state every restart would reset
    Adam's moments to zero and throw away its accumulated per-parameter
    scaling. Written to .tmp then renamed so a crash mid-write can't leave a
    corrupt checkpoint behind."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path, state_path, opt_path = _ckpt_paths(ckpt_dir)
    tmp_w = weights_path.with_name(weights_path.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp_w), dict(tree_flatten(lm.trainable_parameters())))
    tmp_w.replace(weights_path)

    try:
        opt_flat = {k: v for k, v in tree_flatten(opt.state)
                    if isinstance(v, mx.array)}
        if opt_flat:
            tmp_o = opt_path.with_name(opt_path.stem + ".tmp.safetensors")
            mx.save_safetensors(str(tmp_o), opt_flat)
            tmp_o.replace(opt_path)
    except Exception as e:
        # Never let optimizer-state saving kill a run: the params above are
        # already durable, and fresh Adam moments only cost a little re-warmup.
        print(f"[dwq-dist] WARNING: optimizer state not saved ({e})", flush=True)
    # state.json last: its presence is what marks the checkpoint complete.
    state_path.write_text(json.dumps({"step": step, **meta}, indent=2))


def _load_checkpoint(ckpt_dir: Path, lm, opt=None):
    weights_path, state_path, opt_path = _ckpt_paths(ckpt_dir)
    if not (weights_path.exists() and state_path.exists()):
        return 0
    state = json.loads(state_path.read_text())
    saved = mx.load(str(weights_path))
    lm.update(tree_unflatten(list(saved.items())))
    mx.eval(lm.parameters())
    if opt is not None and opt_path.exists():
        try:
            opt.state = tree_unflatten(list(mx.load(str(opt_path)).items()))
            mx.eval(opt.state)
            print(f"[dwq-dist] restored optimizer state from {opt_path}", flush=True)
        except Exception as e:  # a lost optimizer state is recoverable; params are not
            print(f"[dwq-dist] WARNING: could not restore optimizer state ({e}); "
                  "continuing with fresh moments", flush=True)
    return state["step"]


def _block_forward(layer, h, mask, ids):
    """One DeepseekV4Block, matching awq_quantize_deepseek_v4_distributed.py's
    manual reconstruction of DeepseekV4Block.__call__ line-for-line."""
    residual = h
    x, post, comb = layer.attn_hc(h)
    x = layer.attn(layer.attn_norm(x), mask=mask, cache=None)
    h = hc_expand(x, residual, post, comb)

    residual = h
    x, post, comb = layer.ffn_hc(h)
    x = layer.ffn(layer.ffn_norm(x), ids)
    h = hc_expand(x, residual, post, comb)
    return h


def _expand_hc(h, hc_mult):
    h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], hc_mult, h.shape[2]))
    return mx.contiguous(h)


def train_distributed(
    student_path, targets_dir, ckpt_dir, layer_split, lr, scale, max_steps,
    scales_only, optimizer, val_frac, eval_n, train_max_tokens, ckpt_every,
    chunk_layers=1, special_ids=(1,), special_weight=1.0,
):
    group = mx.distributed.init()
    if group is None or group.size() != 2:
        raise SystemExit("this script requires exactly 2 ranks under mlx.launch")
    rank, world_size = group.rank(), group.size()

    _t_start = time.time()
    _patch_switch_stop_grad()
    print(f"[dwq-dist] rank {rank}/{world_size}: loading student (lazy, strict=False): "
          f"{student_path}", flush=True)
    model, processor = load(student_path, lazy=True, strict=False)
    print(f"[dwq-dist] rank {rank}: load() took {time.time()-_t_start:.1f}s", flush=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    text = _text_model(model)
    lm = model.language_model
    lm.train()

    n_layers_total = len(text.layers)
    split_counts = [int(x) for x in layer_split.split(",")]
    if len(split_counts) != world_size or sum(split_counts) != n_layers_total:
        raise ValueError(f"--layer-split must list {world_size} counts summing to "
                          f"{n_layers_total} (got {layer_split!r})")
    # Reversed convention, matching PipelineMixin/the AWQ distributed script:
    # rank 0 = LAST layers. split_counts given in forward order (position 0
    # = first layers).
    position = world_size - rank - 1
    end_idx = sum(split_counts[: position + 1])
    start_idx = end_idx - split_counts[position]
    text.layers = text.layers[:end_idx]
    text.layers[:start_idx] = [None] * start_idx
    mx.eval(model.language_model.parameters())
    hc_mult = text.args.hc_mult
    sliding_window = text.args.sliding_window
    hidden_size = text.args.hidden_size
    compute_dtype = text.embed_tokens.weight.dtype
    print(f"[dwq-dist] rank {rank}: boundary compute_dtype={compute_dtype}", flush=True)
    print(f"[dwq-dist] rank {rank}: owns layers [{start_idx}, {end_idx}) of "
          f"{n_layers_total} (split={split_counts})", flush=True)

    _t_unfreeze = time.time()
    keys = ("scales",) if scales_only else ("scales", "biases")
    n_trainable = _unfreeze_affine_scales(lm, keys)
    n_params = sum(v.size for _, v in tree_flatten(lm.trainable_parameters()))
    print(f"[dwq-dist] rank {rank}: unfroze {n_trainable} local affine modules, "
          f"{n_params/1e6:.1f}M trainable scale params "
          f"(took {time.time()-_t_unfreeze:.1f}s)", flush=True)

    # Map each trainable param's position in the flattened list to the layer it
    # belongs to, so _forward_step can hand a layer's params to mx.checkpoint as
    # EXPLICIT arguments. This is not cosmetic: params captured by closure inside
    # a checkpointed function silently get ZERO gradients while the loss value
    # still looks correct (verified in scratchpad/test_ckpt_grad.py) -- training
    # would appear to run and learn nothing.
    _layer_re = re.compile(r"^model\.layers\.(\d+)\.")
    _flat_keys0 = [k for k, _ in tree_flatten(lm.trainable_parameters())]
    _per_layer: dict[int, list[tuple[int, str]]] = {}
    _other: list[tuple[int, str]] = []
    for _j, _k in enumerate(_flat_keys0):
        _m = _layer_re.match(_k)
        if _m:
            _per_layer.setdefault(int(_m.group(1)), []).append((_j, _k[_m.end():]))
        else:
            _other.append((_j, _k))
    print(f"[dwq-dist] rank {rank}: {len(_per_layer)} layers carry trainable "
          f"params, {len(_other)} non-layer trainable tensors", flush=True)

    files = _load_targets(targets_dir)
    val_stride = max(1, round(1 / val_frac)) if val_frac > 0 else 0
    if val_stride:
        val_files = files[::val_stride]
        files = [f for i, f in enumerate(files) if i % val_stride != 0]
    else:
        val_files = []
    eval_files = val_files[:eval_n] if eval_n else val_files

    meta = json.loads((Path(targets_dir) / "targets_meta.json").read_text())
    collected_max_tokens = int(meta.get("max_tokens", 384))
    pad_len = min(collected_max_tokens, train_max_tokens) if train_max_tokens else collected_max_tokens
    print(f"[dwq-dist] rank {rank}: {len(files)} train / {len(val_files)} held out, "
          f"seq={pad_len} (collected at {collected_max_tokens})", flush=True)
    try:
        _dev = mx.device_info()
        _limit = _dev.get("max_recommended_working_set_size", 0) / 1e9
    except Exception:
        _limit = 0.0
    print(f"[dwq-dist] rank {rank}: after setup active="
          f"{mx.get_active_memory()/1e9:.1f}GB, GPU working-set limit={_limit:.1f}GB",
          flush=True)

    ck_dir = Path(ckpt_dir) / f"rank{rank}"
    opt = optim.SGD(learning_rate=lr) if optimizer == "sgd" else optim.Adam(learning_rate=lr)
    resume_step = _load_checkpoint(ck_dir, lm, opt)
    if resume_step:
        print(f"[dwq-dist] rank {rank}: RESUMING from step {resume_step}", flush=True)
    ckpt_meta = {"student": student_path, "targets": targets_dir, "rank": rank,
                 "start_idx": start_idx, "end_idx": end_idx, "layer_split": split_counts,
                 "scales_only": scales_only, "optimizer": optimizer, "lr": lr}

    # Chunk this rank's layers. Backprop is run one chunk at a time, each in its
    # own mx.vjp + mx.eval, because the peak memory of a quantized-MoE backward
    # is ~26GB PER LAYER and layers inside a single graph stack linearly
    # (measured: 1 layer 26.2GB, 2 layers 52.0GB, 4 layers 103.5GB -- and
    # independent of sequence length). One vjp over 29 layers would need ~750GB.
    # Separate eval'd calls let MLX free each chunk's transients before the next,
    # bounding peak at ~26GB * chunk_layers no matter how deep the stack is.
    #
    # NOTE: mx.checkpoint does NOT help here and is deliberately unused. It only
    # avoids retaining forward activations, but this memory is the dequantized
    # [256, out, in] expert weights + their gradients materialized inside the
    # backward itself. Measured identical to the byte with and without it.
    _chunks = [(a, min(a + chunk_layers, end_idx))
               for a in range(start_idx, end_idx, chunk_layers)]
    print(f"[dwq-dist] rank {rank}: {len(_chunks)} backward chunks of "
          f"<={chunk_layers} layer(s); expected peak ~{26*chunk_layers}GB above weights",
          flush=True)

    def _make_chunk_fn(a, b, mask, inp):
        """Build (fn, global_slot_indices) for layers [a, b).

        fn(h_in, *param_vals) -> h_out, with the chunk's trainable params taken
        as EXPLICIT arguments so mx.vjp differentiates w.r.t. them. Params
        captured by closure would silently yield zero gradients.
        """
        layer_slots = [(i, _per_layer.get(i, [])) for i in range(a, b)]
        counts = [len(s) for _, s in layer_slots]
        gidx = [j for _, slots in layer_slots for j, _ in slots]

        def fn(h_in, *pv):
            h = h_in
            off = 0
            for (i, slots), c in zip(layer_slots, counts):
                if c:
                    sub = pv[off:off + c]
                    text.layers[i].update(
                        tree_unflatten([(rk, v) for (_, rk), v in zip(slots, sub)]))
                off += c
                h = _block_forward(text.layers[i], h, mask, inp)
            return h

        return fn, gidx

    def _forward_chunks(inp, mask, flat_vals, h_in=None):
        """Plain forward over all local chunks, keeping only the small boundary
        h entering each chunk (~25MB each) for the backward to recompute from."""
        if _other:
            lm.update(tree_unflatten([(k, flat_vals[j]) for j, k in _other]))
        h = _expand_hc(text.embed_tokens(inp), hc_mult) if h_in is None else h_in
        boundaries = []
        for (a, b) in _chunks:
            boundaries.append(h)
            fn, gidx = _make_chunk_fn(a, b, mask, inp)
            h = fn(h, *[flat_vals[j] for j in gidx])
            mx.eval(h)
            h = mx.stop_gradient(h)
        return h, boundaries

    def _backward_chunks(inp, mask, flat_vals, boundaries, cot):
        """Reverse-order per-chunk vjp. Returns (grads_by_slot, d_input_h)."""
        grads_out = [None] * len(flat_vals)
        for (a, b), h_in in zip(reversed(_chunks), reversed(boundaries)):
            fn, gidx = _make_chunk_fn(a, b, mask, inp)
            primals = (h_in,) + tuple(flat_vals[j] for j in gidx)
            _, (dh_in, *gp) = mx.vjp(fn, primals, (cot,))
            mx.eval(dh_in, *gp)
            for j, g in zip(gidx, gp):
                grads_out[j] = g
            cot = dh_in
            mx.clear_cache()
        return grads_out, cot

    def _grad_norm_check(grads_flat, verbose):
        """Guard against the silent failure mode where mx.checkpoint returns a
        correct-looking loss but all-zero gradients (happens if params are
        captured by closure rather than passed as explicit args). Zero grads
        mean training runs at full speed and learns nothing, so fail loudly."""
        n_zero = sum(1 for g in grads_flat if not bool(mx.any(g != 0).item()))
        if n_zero == len(grads_flat) and grads_flat:
            raise RuntimeError(
                f"[dwq-dist] rank {rank}: ALL {len(grads_flat)} gradient tensors are "
                "zero -- checkpointing is silently detaching the graph. Aborting "
                "rather than training a no-op.")
        if verbose:
            gn = sum(float((g.astype(mx.float32) ** 2).sum().item()) for g in grads_flat)
            print(f"[dwq-dist] rank {rank}: grad_norm={gn ** 0.5:.4e} "
                  f"({len(grads_flat)-n_zero}/{len(grads_flat)} tensors nonzero)",
                  flush=True)

    def run_step(f, train: bool, verbose: bool = False):
        def mark(label, t_prev):
            if verbose:
                t_now = time.time()
                act = mx.get_active_memory() / 1e9
                peak = mx.get_peak_memory() / 1e9
                print(f"[dwq-dist] rank {rank}: {label} took {t_now-t_prev:.1f}s "
                      f"[active={act:.1f}GB peak={peak:.1f}GB]", flush=True)
                return t_now
            return t_prev

        t = time.time()
        d = mx.load(f)
        t = mark("mx.load(target file)", t)
        inp, idx, vals, tmask, tlogz = _pad(d, pad_len)
        mask = create_attention_mask(
            mx.zeros((inp.shape[0], inp.shape[1], hidden_size)), None,
            window_size=sliding_window, return_array=True,
        )
        t = mark("pad+mask setup", t)
        flat_params = tree_flatten(lm.trainable_parameters())
        flat_keys = [k for k, _ in flat_params]
        flat_vals = [v for _, v in flat_params]
        t = mark("tree_flatten(trainable_parameters)", t)

        if position == 0:  # owns first layers (rank world_size-1, e.g. rank 1 of 2)
            h, boundaries = _forward_chunks(inp, mask, flat_vals)
            t = mark("forward chunks (embed + local layers)", t)
            h_sent = mx.distributed.send(h, 0)
            mx.eval(h_sent)
            t = mark("send(h) + eval", t)

            if not train:
                return None  # eval-only prompts need no backward here

            dh = mx.distributed.recv_like(h, 0)
            mx.eval(dh)
            t = mark("recv(dh) + eval", t)
            grads_flat, _ = _backward_chunks(inp, mask, flat_vals, boundaries, dh)
            t = mark(f"backward {len(_chunks)} chunks", t)
            grads_flat = [g for g in grads_flat if g is not None]
            _grad_norm_check(grads_flat, verbose)
            grads_tree = tree_unflatten(list(zip(flat_keys, grads_flat)))
            opt.update(lm, grads_tree)
            mx.eval(lm.parameters(), opt.state)
            t = mark("opt.update + eval", t)
            return None
        else:  # rank 0: owns last layers + head
            h_shape = (inp.shape[0], inp.shape[1], hc_mult, hidden_size)
            h_in = mx.distributed.recv_like(mx.zeros(h_shape, dtype=compute_dtype), 1)
            mx.eval(h_in)
            t = mark("recv(h_in) + eval", t)

            def head_fn(h_last):
                collapsed = text.norm(text.hc_head(h_last))
                logits = lm.lm_head(collapsed)[0]
                return _kl_topk(logits, idx, vals, tmask, scale, tlogz,
                                special_ids, special_weight)

            h_out, boundaries = _forward_chunks(inp, mask, flat_vals, h_in=h_in)
            t = mark("forward chunks (local layers)", t)

            if not train:
                loss = head_fn(h_out)
                mx.eval(loss)
                return loss.item()

            # Head + loss backward first: gives the cotangent seeding the chunked
            # backward. Head params are 8-bit (frozen), so only d(loss)/d(h) matters.
            (loss,), (dh_out,) = mx.vjp(head_fn, (h_out,), (mx.array(1.0),))
            mx.eval(loss, dh_out)
            t = mark("head+loss vjp", t)

            grads_flat, dh_input = _backward_chunks(
                inp, mask, flat_vals, boundaries, dh_out)
            t = mark(f"backward {len(_chunks)} chunks", t)
            dh_sent = mx.distributed.send(dh_input, 1)
            mx.eval(dh_sent)
            t = mark("send(dh) + eval", t)
            grads_flat = [g for g in grads_flat if g is not None]
            _grad_norm_check(grads_flat, verbose)
            grads_tree = tree_unflatten(list(zip(flat_keys, grads_flat)))
            opt.update(lm, grads_tree)
            mx.eval(lm.parameters(), opt.state)
            t = mark("opt.update + eval", t)
            return loss.item()

    def eval_held_out(label):
        losses = []
        for f in eval_files:
            loss = run_step(f, train=False)
            if loss is not None:
                losses.append(loss)
        if rank == 0 and losses:
            avg = sum(losses) / len(losses)
            print(f"[dwq-dist] rank0 HELD-OUT kl at {label}: {avg:.4f} (n={len(losses)})", flush=True)

    if eval_files:
        eval_held_out("step 0 (pre-training baseline)")

    t0 = time.time()
    step = 0
    for f in files:
        step += 1
        if step <= resume_step:
            continue
        ts = time.time()
        loss = run_step(f, train=True, verbose=(step <= 2))
        if rank == 0 and (step <= 3 or step % 10 == 0):
            print(f"[dwq-dist] step {step}/{len(files)} kl={loss:.4f} "
                  f"({time.time()-ts:.1f}s)", flush=True)
        if step % ckpt_every == 0:
            _save_checkpoint(ck_dir, lm, opt, step, ckpt_meta)
            if rank == 0:
                print(f"[dwq-dist] checkpoint saved at step {step}", flush=True)
            if eval_files:
                eval_held_out(f"step {step}")
        if max_steps and step >= max_steps:
            print(f"[dwq-dist] rank {rank}: hit max_steps={max_steps}, stopping", flush=True)
            break

    _save_checkpoint(ck_dir, lm, opt, step, ckpt_meta)
    if eval_files:
        eval_held_out(f"step {step} (final)")
    print(f"[dwq-dist] rank {rank}: done in {time.time()-t0:.0f}s, "
          f"checkpoint -> {ck_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--layer-split", required=True,
                    help="Comma-separated layer counts, forward order (position 0 = "
                         "first layers), summing to the model's total layer count.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--scales-only", action="store_true")
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-n", type=int, default=30)
    ap.add_argument("--train-max-tokens", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--special-weight", type=float, default=1.0,
                    help="Weight on the one-sided EOS/special-token bound.")
    ap.add_argument("--special-ids", default="1",
                    help="Comma-separated token ids to bound (default EOS=1).")
    ap.add_argument("--chunk-layers", type=int, default=1,
                    help="Layers per backward chunk. Peak memory is ~26GB per "
                         "layer in flight, so 1 is the safe default; raise only "
                         "if the rank has spare headroom.")
    a = ap.parse_args()
    train_distributed(
        a.student, a.targets, a.ckpt_dir, a.layer_split, a.lr, a.scale,
        a.max_steps, a.scales_only, a.optimizer, a.val_frac, a.eval_n,
        a.train_max_tokens, a.ckpt_every, a.chunk_layers,
        tuple(int(x) for x in a.special_ids.split(',') if x.strip()), a.special_weight,
    )


if __name__ == "__main__":
    main()
