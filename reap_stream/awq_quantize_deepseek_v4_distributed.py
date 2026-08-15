"""Pipeline-parallel variant of awq_quantize_deepseek_v4.py -- splits the 43
decoder layers across 2+ machines (e.g. M5 Max + M3 Max over a Thunderbolt
bridge) using mlx_vlm's built-in DeepSeek-V4 pipeline-parallel support
(PipelineMixin / sharded_load), rather than tensor parallelism (shard()),
which needs per-layer all-reduce/all-gather synchronization -- pipeline
parallelism needs only ONE cross-machine handoff per forward pass, which
matters a lot more on a 2-node consumer link than it would on real NVLink.

REAL, VERIFIED-EXISTING infrastructure this builds on (read directly from
the installed mlx_vlm source, not assumed):
- mlx_vlm.utils.sharded_load(repo, pipeline_group=...) truncates
  text.layers to this rank's local slice and sets
  start_idx/end_idx/pipeline_rank/pipeline_size (mlx_lm.models.pipeline.
  PipelineMixin.pipeline()).
- Layer assignment is REVERSED: pipeline_rank=0 gets the LAST layers,
  pipeline_rank=pipeline_size-1 gets the FIRST layers. Confirmed in
  PipelineMixin.pipeline()'s own comment and index math.
- DeepseekV4Model.__call__'s own forward pass already does the cross-rank
  handoff (mx.distributed.recv_like before its local layers if
  pipeline_rank < pipeline_size-1, mx.distributed.send after if
  pipeline_rank != 0) -- but ONLY when going through that top-level call.
  This driver bypasses it (same reason as the single-machine version:
  needs to intercept ffn_norm's output for AWQ calibration), so it has to
  replicate that exact recv/send protocol itself, at the boundary of
  its own locally-owned layer range.

WHAT IS NOT YET VERIFIED (no second machine to test against as of writing):
the actual 2-rank recv/send handoff below, end to end. The single-rank
path (world_size==1) is unchanged from awq_quantize_deepseek_v4.py and
behaves identically -- that IS tested. The distributed path is a careful,
protocol-matched implementation, not a proven one. A mismatched send/recv
pairing would deadlock rather than error loudly, so the first real run
should be watched, not launched and left.

Launch (once the second machine is connected, SSH key-authed, same MLX/
mlx_vlm versions installed):
    mlx.launch --hosts <m5max-ip>,<m3max-ip> --backend ring -- \
        .venv/bin/python -m reap_stream.awq_quantize_deepseek_v4_distributed \
        --model ~/Desktop/models/DeepSeek-V4-Flash-0731 \
        --dataset calib/cloud_reap_8k.jsonl \
        --out models/DeepSeek-V4-Flash-0731-awq2bit3bit \
        --n-prompts 128 --max-tokens 384 --bits 2 --down-bits 3 --group-size 128 --n-grid 20

Single-machine fallback (world_size==1, no mlx.launch): identical to
awq_quantize_deepseek_v4.py.

--- Original single-machine docstring below ---

AWQ-calibrated quantization for DeepSeek-V4-Flash's routed experts only
(switch_mlp gate_proj/up_proj/down_proj) -- a narrower, purpose-built driver
rather than a direct call into mlx_lm.quant.awq.awq_quantize, for two real
reasons verified against the installed mlx_vlm source:

1. awq_quantize's outer loop assumes a plain decoder block: `block(inputs,
   mask=mask)` returning a tensor of the same (batch, seq, hidden) shape to
   feed the next layer. DeepSeek-V4 breaks this twice over -- hidden state
   is (batch, seq, hc_mult, hidden) (Hyper-Connections' 4 parallel residual
   streams, only collapsed at the very end via hc_head), and every block
   needs `input_ids` threaded in for hash-routed layers' fixed token->expert
   lookup (layer(h, mask, cache, input_ids), not layer(x, mask=mask)).
2. awq_quantize's whole-block quantize/measure/revert cycle would touch
   attention, compressor, indexer, and shared_experts too -- all of which
   this project's build policy (build_deepseek_v4_quant98.py) deliberately
   leaves at native mxfp4/mxfp8 precision. Scoping AWQ to just
   ffn.switch_mlp avoids ever touching those modules, rather than trying to
   exclude them after the fact via clip_block's substring-matching
   no_clip_keys (fragile: "gate" would need to distinguish the router's
   ffn.gate from switch_mlp.gate_proj).

Reused as-is from mlx_lm.quant.awq (verified generic, no changes needed):
search_best_scale, search_best_clip, apply_scale, submodule_from_key.

Usage:
    .venv/bin/python -m reap_stream.awq_quantize_deepseek_v4 \
        --model ~/Desktop/models/DeepSeek-V4-Flash-0731 \
        --dataset calib/cloud_reap_8k.jsonl \
        --out models/DeepSeek-V4-Flash-0731-awq2bit3bit \
        --n-prompts 128 --max-tokens 384 --bits 2 --down-bits 3 --group-size 128
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
from mlx_vlm.utils import sharded_load
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.switch_layers import SwitchGLU, SwitchLinear, QuantizedSwitchLinear
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
from mlx.utils import tree_flatten
import mlx_lm.quant.awq as _awq_mod
from mlx_lm.quant.awq import apply_scale, mse

# mlx_lm.quant.awq's isinstance(x, (nn.Linear, SwitchLinear)) checks (inside
# search_best_scale's perturbation loop, search_best_clip, and apply_scale)
# all reference mlx_lm.models.switch_layers.SwitchLinear specifically.
# DeepSeek-V4 only exists via mlx_vlm, whose SwitchLinear is a structurally
# identical but DIFFERENT class -- confirmed the hard way: apply_scale raised
# NotImplementedError for an mlx_vlm SwitchLinear instance ("Could not apply
# scale to prev_op"), and search_best_scale's matching check would have
# silently no-op'd the weight-perturbation step instead of erroring (worse:
# silently meaningless scales, not a loud failure). Patching the module-level
# name fixes every isinstance check in that module at once, since they all
# share the same globals -- safer than hunting down each call site
# individually and assuming none were missed.
_awq_mod.SwitchLinear = SwitchLinear

# run_layer's default batch_size=32 micro-batches x for memory safety, but
# only slices x/indices -- any OTHER kwarg (our "input_ids") stays at full
# size across every micro-batch. Invisible at n_prompts<=32 (the loop runs
# once, no real slicing happens); confirmed broken the hard way at
# n_prompts=128: ValueError broadcasting (32,384) input against (128,384)
# input_ids inside the router. Can't fix by adding batch_size to
# layer_kwargs either -- that dict is ALSO used in a direct (non-run_layer)
# call to layer.ffn(x, **layer_kwargs), whose __call__ doesn't accept a
# batch_size argument. Raising the default here instead avoids
# micro-batching altogether for realistic prompt counts (forward-only, no
# gradients retained, so the full batch fits comfortably in memory).
_orig_run_layer = _awq_mod.run_layer
def _run_layer_no_microbatch(layer, x, indices=None, batch_size=4096, **kwargs):
    return _orig_run_layer(layer, x, indices=indices, batch_size=batch_size, **kwargs)
_awq_mod.run_layer = _run_layer_no_microbatch
run_layer = _run_layer_no_microbatch

# Forked from mlx_lm.quant.awq.search_best_scale/search_best_clip with the
# `group = mx.distributed.init(); if group is not None: loss = all_sum(loss)
# / group.size()` lines removed. Those calls assume DATA-parallel semantics
# -- multiple ranks redundantly computing the SAME objective on different
# data shards, needing their losses averaged. We're PIPELINE-parallel: rank
# 0 and rank 1 are calibrating completely different, independent layers at
# any given moment, so there is nothing to average, and no reason for them
# to synchronize on it. Confirmed the hard way: with the original functions,
# search_best_scale's mx.eval(loss) hit a real Metal GPU command-buffer
# timeout under mlx.launch's 2-rank ring -- the all_sum was trying to
# rendezvous with a rank that had no matching call to make (it was off
# processing its own unrelated layer), not a perf issue fixable by waiting
# longer.
def search_best_scale(layers, quantize_func, block, layer_kwargs, n_grid):
    layer_kwargs = layer_kwargs or {}
    x = layers[0].input_feat
    block = block or layers[0]
    out = block(x, **layer_kwargs)
    x_max = x.abs().mean(axis=(0, 1))
    best_error = float("inf")
    best_scales = None
    weights = tree_flatten(block.parameters())
    for ratio in range(n_grid):
        ratio = ratio / n_grid
        scales = mx.maximum(x_max**ratio, 1e-4).reshape(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for layer in layers:
            if isinstance(layer, (nn.Linear, SwitchLinear)):
                layer.weight = quantize_func(layer.weight * scales) / scales
        out_q = run_layer(block, x, **layer_kwargs)
        loss = mse(out, out_q).sum()
        loss /= out.size
        mx.eval(loss)
        if loss.item() < best_error:
            best_error = loss.item()
            best_scales = scales
        block.load_weights(weights)
    best_scales = best_scales.reshape(-1)
    mx.eval(best_scales)
    return best_scales


def search_best_clip(module, quantize_func, group_size, n_grid, max_shrink=0.5, batch_size=64, n_frames=512):
    x = module.input_feat.flatten(0, 1)
    stride = (x.shape[0] + n_frames - 1) // n_frames
    x = x[::stride]
    w = module.weight
    x = x.reshape(x.shape[0], -1, group_size)
    w_init_shape = w.shape
    w_all = mx.flatten(w, 0, w.ndim - 2)
    w_max_all = []
    for b in range(0, w_all.shape[0], batch_size):
        w = w_all[b : b + batch_size]
        group_shape = (w.shape[0], w.shape[-1] // group_size)
        best_error = mx.full(group_shape, float("inf"))
        best_w_max = mx.zeros((*group_shape, 1), dtype=x.dtype)
        w_shape = w.shape
        w = w.reshape(*w.shape[:-1], -1, group_size)
        out = mx.einsum("bdg,odg->bod", x, w)
        init_max = w.abs().max(axis=-1, keepdims=True)
        for i in range(int(max_shrink * n_grid)):
            p = 1 - i / n_grid
            w_max = p * init_max
            w_m = mx.clip(w, -w_max, w_max).reshape(w_shape)
            w_q = quantize_func(w_m)
            w_q = w_q.reshape(*w_q.shape[:-1], -1, group_size)
            out_q = mx.einsum("bdg,odg->bod", x, w_q)
            loss = mse(out, out_q).sum(axis=0)
            loss /= out.shape[0]
            best_indices = loss < best_error
            best_error = mx.where(best_indices, loss, best_error)
            best_w_max = mx.where(best_indices[..., mx.newaxis], w_max, best_w_max)
            mx.eval(best_w_max, best_error)
        w_max_all.append(best_w_max)
    best_w_max = mx.concatenate(w_max_all, axis=0)
    w_r = w_all.reshape(*w_all.shape[:-1], -1, group_size)
    best_w = mx.clip(w_r, -best_w_max, best_w_max)
    best_w = best_w.reshape(w_init_shape)
    mx.eval(best_w)
    return best_w


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _load_filtered_prompts(dataset_file: str, limit: int, exclude_categories: set[str]) -> list[str]:
    out = []
    with open(dataset_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("category") in exclude_categories:
                continue
            text = rec.get("text")
            if text and str(text).strip():
                out.append(str(text).strip())
            if len(out) >= limit:
                break
    return out


def _tokenize_prompts(tokenizer, prompts: list[str], max_tokens: int) -> list[list[int]]:
    batches = []
    for p in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text_in = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text_in = p
        else:
            text_in = p
        tokens = tokenizer.encode(text_in)[:max_tokens]
        batches.append(tokens)
    return batches


def _expand_hc(h: mx.array, hc_mult: int) -> mx.array:
    h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], hc_mult, h.shape[2]))
    return mx.contiguous(h)


def _layer_ckpt_path(ckpt_dir: str, i: int) -> Path:
    return Path(ckpt_dir) / f"layer_{i:03d}.safetensors"


def _save_layer_ckpt(ckpt_dir: str, i: int, sg, bits: int, down_bits: int, group_size: int) -> None:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        mod = getattr(sg, name)
        tensors[f"{name}.weight"] = mod.weight
        tensors[f"{name}.scales"] = mod.scales
        b = mod.get("biases")
        if b is not None:
            tensors[f"{name}.biases"] = b
    mx.eval(list(tensors.values()))
    p = _layer_ckpt_path(ckpt_dir, i)
    tmp = p.with_suffix(".tmp.safetensors")
    mx.save_safetensors(str(tmp), tensors)
    tmp.replace(p)


def _load_layer_ckpt(ckpt_dir: str, i: int, sg, bits: int, down_bits: int, group_size: int) -> bool:
    """Reconstruct QuantizedSwitchLinear modules from a saved layer checkpoint,
    reading input_dims/output_dims/num_experts off the CURRENT (still-native,
    not yet dequantized) sg modules before overwriting them."""
    p = _layer_ckpt_path(ckpt_dir, i)
    if not p.exists():
        return False
    t = mx.load(str(p))
    for name, bits_for in (("gate_proj", bits), ("up_proj", bits), ("down_proj", down_bits)):
        orig = getattr(sg, name)
        q = QuantizedSwitchLinear(
            orig.input_dims, orig.output_dims, orig.num_experts,
            False, group_size, bits_for, mode="affine",
        )
        q.weight = t[f"{name}.weight"]
        q.scales = t[f"{name}.scales"]
        q.biases = t.get(f"{name}.biases")
        setattr(sg, name, q)
    mx.eval(sg.gate_proj, sg.up_proj, sg.down_proj)
    return True


def _dequant_to_switch_linear(qmod) -> SwitchLinear:
    """QuantizedSwitchLinear (native mxfp4) -> plain float SwitchLinear.
    Required before running AWQ: apply_scale/search_best_scale/search_best_clip
    all assume plain (experts, out, in) float weights they can multiply/divide
    by per-channel scales -- confirmed the hard way (ValueError: shapes
    (256,2048,512) and (4096,) cannot be broadcast, 512 being the packed
    4-bit width, not a real dimension). No bias handling needed: SwitchGLU
    constructs its three projections with bias=False (DeepseekV4MoE never
    passes bias=True), verified against the installed source."""
    full = mx.dequantize(
        qmod.weight, qmod.scales, qmod.get("biases"),
        group_size=qmod.group_size, bits=qmod.bits, mode=qmod.mode,
    )
    mx.eval(full)
    plain = SwitchLinear(qmod.input_dims, qmod.output_dims, qmod.num_experts, bias=False)
    plain.weight = full
    return plain


class _SwitchCatcher(nn.Module):
    """Wraps a SwitchGLU projection (gate_proj/up_proj/down_proj) to record
    its real input features and MoE routing indices during the reference
    forward pass, matching mlx_lm.quant.awq's own Catcher contract (the
    .input_feat / .indices attributes search_best_scale/clip_block expect)."""

    def __init__(self, inner):
        super().__init__()
        self.module = inner

    def __call__(self, x, indices, *args, **kwargs):
        if hasattr(self, "input_feat"):
            self.input_feat = mx.concatenate([self.input_feat, x], axis=0)
        else:
            self.input_feat = x
        if hasattr(self, "indices"):
            self.indices = mx.concatenate([self.indices, indices], axis=0)
        else:
            self.indices = indices
        return self.module(x, indices, *args, **kwargs)


def _quantize_func(bits: int, group_size: int):
    def f(w):
        wq = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(*wq, bits=bits, group_size=group_size)
    return f


def awq_quantize_experts(
    model_path: str,
    dataset_file: str,
    out_dir: str,
    n_prompts: int,
    max_tokens: int,
    bits: int,
    down_bits: int,
    group_size: int,
    n_grid: int,
    exclude_categories: list[str],
    ckpt_dir: str | None = None,
    layer_split: str | None = None,
):
    out = Path(out_dir)
    t0 = time.time()
    if ckpt_dir is None:
        ckpt_dir = str(out) + "-ckpt"
    print(f"[awq-dsv4] checkpointing per-layer to {ckpt_dir}", flush=True)

    group = mx.distributed.init()
    world_size = group.size()
    rank = group.rank()
    distributed = world_size > 1

    if distributed:
        # NOT sharded_load(pipeline_group=group): that calls PipelineMixin.
        # pipeline(), which only does an EVEN split (len(layers)//world_size).
        # Real machines here have very different RAM (M5 Max 128GB vs M3 Max
        # 68.7GB) -- confirmed the hard way: an even 21/22 split put 88.1GB
        # of shards on the M3 Max, which OOM-killed silently (exit 255, no
        # traceback) during materialization, since 88GB > its 68.7GB total
        # RAM. Replicating sharded_load's own steps manually here, swapping
        # only the split logic for one driven by --layer-split.
        from mlx_vlm.utils import load_model, load_processor, load_image_processor
        print(f"[awq-dsv4] rank {rank}/{world_size}: loading (lazy, strict=False): "
              f"{model_path}", flush=True)
        model_path_p = Path(model_path)
        model = load_model(model_path_p, lazy=True, strict=False)
        config = model.config.to_dict()
        processor = load_processor(model_path_p, True, eos_token_ids=config.get("eos_token_id", None))
        image_processor = load_image_processor(model_path_p)
        if image_processor is not None:
            processor.image_processor = image_processor

        lm = model.language_model
        inner = lm.model if hasattr(lm, "model") else lm
        n_layers_total = len(inner.layers)
        split_counts = [int(x) for x in layer_split.split(",")] if layer_split else None
        if split_counts is None or len(split_counts) != world_size or sum(split_counts) != n_layers_total:
            raise ValueError(
                f"--layer-split must list {world_size} counts summing to "
                f"{n_layers_total} (got {layer_split!r})"
            )
        # Same reversed convention as PipelineMixin.pipeline(): rank 0 = LAST
        # layers. split_counts is given in forward layer order (position 0 =
        # first layers), so rank r's position is (world_size - r - 1).
        position = world_size - rank - 1
        end_idx = sum(split_counts[: position + 1])
        start_idx = end_idx - split_counts[position]
        inner.pipeline_rank = rank
        inner.pipeline_size = world_size
        inner.start_idx = start_idx
        inner.end_idx = end_idx
        inner.layers = inner.layers[:end_idx]
        inner.layers[:start_idx] = [None] * start_idx

        mx.eval(model.language_model.parameters())
        model.eval()
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
        print(f"[awq-dsv4] rank {rank}: owns layers [{start_idx}, {end_idx}) "
              f"of {n_layers_total} (custom split={split_counts})", flush=True)
    else:
        print(f"[awq-dsv4] loading (lazy): {model_path}", flush=True)
        model, processor = load(model_path, lazy=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    text = _text_model(model)
    if not distributed:
        n_layers_total = text.args.num_hidden_layers
        start_idx, end_idx = 0, n_layers_total
    hc_mult = text.args.hc_mult
    sliding_window = text.args.sliding_window

    prompts = _load_filtered_prompts(dataset_file, n_prompts, set(exclude_categories))
    token_batches = _tokenize_prompts(tokenizer, prompts, max_tokens)
    print(f"[awq-dsv4] {len(token_batches)} prompts, {n_layers_total} layers "
          f"total ({end_idx - start_idx} local), "
          f"bits={bits} (down={down_bits}), group_size={group_size}", flush=True)

    # Fixed-length right-padding: search_best_scale/search_best_clip concatenate
    # every calibration batch's captured activations into ONE tensor (axis=0),
    # which requires matching shape on every other axis -- real prompts here
    # vary in length (confirmed: a 5-token prompt vs a 384-token one), so
    # without padding the concatenate itself would fail on shape mismatch.
    # Padding tokens contribute some noise to the activation-magnitude scale
    # search (no loss-masking here, unlike DWQ's KL training loss) -- an
    # accepted approximation for a calibration heuristic, not a correctness
    # requirement the way it is for DWQ's actual gradient signal.
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
    pad_len = max_tokens
    hidden, input_ids_list, masks = [], [], []
    for tokens in token_batches:
        padded = list(tokens) + [pad_id] * (pad_len - len(tokens))
        ids = mx.array(padded)[None]
        h = _expand_hc(text.embed_tokens(ids), hc_mult)
        mask = create_attention_mask(h[:, :, 0, :], None, window_size=sliding_window, return_array=True)
        mx.eval(h, mask)
        hidden.append(h)
        input_ids_list.append(ids)
        masks.append(mask)
    all_ids = mx.concatenate(input_ids_list, axis=0)
    mx.eval(all_ids)

    # Cross-rank boundary conditions, matching DeepseekV4Model.__call__'s own
    # protocol exactly (reversed pipeline order: rank 0 = last layers):
    #   receive incoming hidden state from rank+1 before this rank's FIRST
    #   local layer, unless this rank owns the globally-first layers
    #   (rank == world_size-1, start_idx == 0);
    #   send this rank's output to rank-1 after its LAST local layer, unless
    #   this rank owns the globally-last layers (rank == 0).
    needs_recv = distributed and rank < world_size - 1
    needs_send = distributed and rank != 0

    for i in range(start_idx, end_idx):
        layer = text.layers[i]
        sg = layer.ffn.switch_mlp

        if ckpt_dir and _load_layer_ckpt(ckpt_dir, i, sg, bits, down_bits, group_size):
            # Resumed from checkpoint -- skip the expensive search/clip, but
            # still need this layer's real (now AWQ-quantized) output to
            # feed the next layer's calibration input.
            new_hidden = []
            for bi, (h, mask, ids) in enumerate(zip(hidden, masks, input_ids_list)):
                if needs_recv and i == start_idx:
                    h = mx.distributed.recv_like(h, (rank + 1))
                    mx.eval(h)
                h = layer(h, mask, None, ids)
                if needs_send and i == end_idx - 1:
                    h = mx.distributed.send(h, (rank - 1) % world_size)
                new_hidden.append(h)
            mx.eval(new_hidden)
            hidden = new_hidden
            gc.collect()
            mx.clear_cache()
            print(f"[awq-dsv4] layer {i:02d} [{start_idx},{end_idx}) loaded from "
                  f"checkpoint, skipped ({time.time() - t0:.0f}s)", flush=True)
            continue

        # Dequantize native mxfp4 -> plain float SwitchLinear first; AWQ's
        # scale-search/apply_scale/clip all operate on plain float weights.
        sg.gate_proj = _dequant_to_switch_linear(sg.gate_proj)
        sg.up_proj = _dequant_to_switch_linear(sg.up_proj)
        sg.down_proj = _dequant_to_switch_linear(sg.down_proj)

        # 1. Reference forward through the WHOLE block (attn+HC unchanged),
        #    capturing the RAW ffn input (ffn_norm(ffn_hc(h)[0]), shape
        #    (B, L, hidden)) directly at the point DeepseekV4Block.__call__
        #    computes it -- verified empirically that feeding gate_proj's
        #    OWN captured input back into layer.ffn(x, input_ids) fails
        #    (ValueError: shape (1,5,6,1,2048) vs (4096,256)), because
        #    SwitchGLU internally expand_dims + gather/sorts x before
        #    gate_proj/up_proj ever see it. This is the raw pre-gather value
        #    instead, matching DeepseekV4MoE.__call__'s real x parameter.
        ffn_inputs = []
        down_catch = _SwitchCatcher(sg.down_proj)
        sg.down_proj = down_catch

        new_hidden = []
        for h, mask, ids in zip(hidden, masks, input_ids_list):
            if needs_recv and i == start_idx:
                h = mx.distributed.recv_like(h, (rank + 1))
                mx.eval(h)

            residual = h
            x, post, comb = layer.attn_hc(h)
            x = layer.attn(layer.attn_norm(x), mask=mask, cache=None)
            h = hc_expand(x, residual, post, comb)

            residual = h
            x, post, comb = layer.ffn_hc(h)
            ffn_in = layer.ffn_norm(x)
            mx.eval(ffn_in)
            ffn_inputs.append(ffn_in)

            x = layer.ffn(ffn_in, ids)
            h = hc_expand(x, residual, post, comb)

            if needs_send and i == end_idx - 1:
                h = mx.distributed.send(h, (rank - 1) % world_size)
            mx.eval(h)
            new_hidden.append(h)
        hidden = new_hidden

        sg.down_proj = down_catch.module
        sg.down_proj.input_feat = down_catch.input_feat
        sg.down_proj.indices = down_catch.indices

        raw_ffn_input = mx.concatenate(ffn_inputs, axis=0)
        mx.eval(raw_ffn_input)

        # 2. AWQ scale search: ffn_norm -> {gate_proj, up_proj}. Manually
        #    seed .input_feat with the RAW ffn input (not what a leaf-level
        #    catcher on gate_proj would see) since search_best_scale reads
        #    layers[0].input_feat directly and calls block(x, **kwargs).
        qf = _quantize_func(bits, group_size)
        # gate_proj and up_proj both receive the identical raw ffn input
        # inside SwitchGLU.__call__ (same x, two different weight matrices)
        # -- search_best_scale only reads layers[0].input_feat, but
        # search_best_clip is called on up_proj separately later and needs
        # its own .input_feat set too.
        sg.gate_proj.input_feat = raw_ffn_input
        sg.up_proj.input_feat = raw_ffn_input
        scales_gu = search_best_scale(
            layers=[sg.gate_proj, sg.up_proj],
            block=layer.ffn,
            layer_kwargs={"input_ids": all_ids},
            quantize_func=qf,
            n_grid=n_grid,
        )
        apply_scale(layer.ffn_norm, [sg.gate_proj, sg.up_proj], scales_gu)

        # 3. AWQ scale search: up_proj -> down_proj. down_proj's own capture
        #    (via the leaf-level _SwitchCatcher above) IS in the right
        #    layout here -- we're calling down_proj directly (block=None),
        #    not replaying through the whole ffn, so no mismatch.
        qf_down = _quantize_func(down_bits, group_size)
        scales_d = search_best_scale(
            layers=[sg.down_proj],
            block=None,
            layer_kwargs={"indices": sg.down_proj.indices},
            quantize_func=qf_down,
            n_grid=n_grid,
        )
        apply_scale(sg.up_proj, [sg.down_proj], scales_d)

        # 4. Clip search (per-module, same weighting as clip_block but
        #    scoped to just these three -- never touches attn/compressor/
        #    indexer/shared_experts/router).
        sg.gate_proj.weight = search_best_clip(sg.gate_proj, qf, group_size, n_grid)
        sg.up_proj.weight = search_best_clip(sg.up_proj, qf, group_size, n_grid)
        sg.down_proj.weight = search_best_clip(sg.down_proj, qf_down, group_size, n_grid)

        # 5. Final RTN quantize with the AWQ-calibrated scale/clip applied.
        gate_q = sg.gate_proj.to_quantized(group_size=group_size, bits=bits)
        up_q = sg.up_proj.to_quantized(group_size=group_size, bits=bits)
        down_q = sg.down_proj.to_quantized(group_size=group_size, bits=down_bits)
        sg.gate_proj, sg.up_proj, sg.down_proj = gate_q, up_q, down_q
        mx.eval(sg.gate_proj, sg.up_proj, sg.down_proj)

        del raw_ffn_input, ffn_inputs, scales_gu, scales_d
        del gate_q, up_q, down_q
        for h in hidden:
            mx.eval(h)

        if ckpt_dir:
            _save_layer_ckpt(ckpt_dir, i, sg, bits, down_bits, group_size)

        gc.collect()
        mx.clear_cache()
        print(f"[awq-dsv4] rank {rank}: layer {i:02d}/{n_layers_total - 1} "
              f"done ({time.time() - t0:.0f}s)", flush=True)

    if distributed:
        # No single rank holds the full model after pipeline splitting --
        # each rank only has its own layer range resident. Each rank's
        # per-layer checkpoints (already written above) are the real
        # output of this run; assembling them into one servable checkpoint
        # is a separate, single-machine step: copy both ranks'
        # --ckpt-dir contents into one directory, then run
        # awq_quantize_deepseek_v4.py's non-distributed path pointed at
        # that merged ckpt-dir (its checkpoint-resume logic already skips
        # any layer with an existing checkpoint file, so it will load every
        # layer from disk instead of recomputing anything) to do the
        # final save + config write on one machine with the full model.
        print(f"[awq-dsv4] rank {rank}: done with local layers "
              f"[{start_idx},{end_idx}). Per-layer checkpoints are in "
              f"{ckpt_dir} -- merge both ranks' checkpoint directories and "
              f"run the non-distributed script once to assemble the final "
              f"checkpoint ({time.time() - t0:.0f}s)", flush=True)
        return

    print(f"[awq-dsv4] saving -> {out}", flush=True)
    from mlx_vlm.utils import save_weights
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    save_weights(out, model, donate_weights=True)

    src = Path(model_path)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        p = src / name
        if p.exists():
            shutil.copy2(p, out / name)
    template_src = Path(__file__).parent / "assets" / "deepseek_v4_chat_template.jinja"
    if template_src.exists():
        shutil.copy2(template_src, out / "chat_template.jinja")

    from mlx_vlm.models.deepseek_v4.language import make_quantization_config
    quant_cfg = make_quantization_config(model)
    prefix = "language_model.model" if hasattr(model, "language_model") else "model"
    for i in range(n_layers_total):
        quant_cfg[f"{prefix}.layers.{i}.ffn.switch_mlp.gate_proj"] = {"bits": bits, "group_size": group_size, "mode": "affine"}
        quant_cfg[f"{prefix}.layers.{i}.ffn.switch_mlp.up_proj"] = {"bits": bits, "group_size": group_size, "mode": "affine"}
        quant_cfg[f"{prefix}.layers.{i}.ffn.switch_mlp.down_proj"] = {"bits": down_bits, "group_size": group_size, "mode": "affine"}

    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization"] = quant_cfg
    cfg["quantization_config"] = quant_cfg
    cfg["vision_config"] = {}
    cfg["_build_note"] = (
        f"AWQ-calibrated (not plain RTN): switch_mlp gate_proj/up_proj at "
        f"{bits}-bit, down_proj at {down_bits}-bit, group_size={group_size}. "
        f"Attention/shared_experts/router/embed/lm_head left native. "
        f"Calibrated on {len(token_batches)} prompts from {dataset_file}."
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[awq-dsv4] done in {time.time() - t0:.0f}s -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--down-bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-grid", type=int, default=20)
    ap.add_argument("--exclude-categories", nargs="*", default=["multimodal"])
    ap.add_argument("--ckpt-dir", default=None,
                    help="Per-layer resumable checkpoints. Defaults to <out>-ckpt. "
                         "Rerunning with the same --out/--ckpt-dir skips already-"
                         "completed layers instead of reprocessing them.")
    ap.add_argument("--layer-split", default=None,
                    help="Comma-separated layer counts in forward order (position 0 "
                         "= first layers), one per rank, summing to the total layer "
                         "count. Required when world_size > 1 -- there is no default "
                         "even split, since real machines can have very different "
                         "RAM. E.g. '8,35' for a 2-rank run where rank 1 (owns the "
                         "first layers per the reversed pipeline convention) gets 8.")
    a = ap.parse_args()
    awq_quantize_experts(a.model, a.dataset, a.out, a.n_prompts, a.max_tokens,
                          a.bits, a.down_bits, a.group_size, a.n_grid,
                          a.exclude_categories, a.ckpt_dir, a.layer_split)


if __name__ == "__main__":
    main()
