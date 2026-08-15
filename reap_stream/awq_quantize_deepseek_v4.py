"""AWQ-calibrated quantization for DeepSeek-V4-Flash's routed experts only
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
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.switch_layers import SwitchGLU, SwitchLinear, QuantizedSwitchLinear
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
import mlx_lm.quant.awq as _awq_mod
from mlx_lm.quant.awq import (
    apply_scale,
    search_best_clip,
    search_best_scale,
)

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


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _load_filtered_prompts(
    dataset_file: str, limit: int, exclude_categories: set[str]
) -> list[tuple[str, bool]]:
    """Returns (text, prerendered) pairs. prerendered=True means text is
    already a fully chat-template-rendered string (real special tokens
    included) and must NOT be wrapped again via apply_chat_template."""
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
                out.append((str(text).strip(), bool(rec.get("prerendered", False))))
            if len(out) >= limit:
                break
    return out


def _tokenize_prompts(
    tokenizer, prompts: list[tuple[str, bool]], max_tokens: int
) -> list[list[int]]:
    batches = []
    for p, prerendered in prompts:
        if prerendered:
            text_in = p
        elif hasattr(tokenizer, "apply_chat_template"):
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


def _load_layer_ckpt(
    ckpt_dir: str, i: int, sg, bits: int, down_bits: int, group_size: int, down_group_size: int
) -> bool:
    """Reconstruct QuantizedSwitchLinear modules from a saved layer checkpoint,
    reading input_dims/output_dims/num_experts off the CURRENT (still-native,
    not yet dequantized) sg modules before overwriting them."""
    p = _layer_ckpt_path(ckpt_dir, i)
    if not p.exists():
        return False
    t = mx.load(str(p))
    for name, bits_for, gs_for in (
        ("gate_proj", bits, group_size),
        ("up_proj", bits, group_size),
        ("down_proj", down_bits, down_group_size),
    ):
        orig = getattr(sg, name)
        q = QuantizedSwitchLinear(
            orig.input_dims, orig.output_dims, orig.num_experts,
            False, gs_for, bits_for, mode="affine",
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


def _valid_index(lengths: list[int], pad_len: int) -> mx.array:
    """Flat indices of REAL (non-pad) positions in a (P, pad_len, ...) stack."""
    idx = [p * pad_len + t for p, n in enumerate(lengths) for t in range(n)]
    return mx.array(idx, dtype=mx.uint32)


def _drop_pad(x: mx.array, flat_idx: mx.array) -> mx.array:
    """Gather real positions out of a (P, L, ...) tensor -> (1, N, ...).

    AWQ's scale search reduces with x.abs().mean(axis=(0, 1)) and scores
    mse(out, out_q) over every captured position, so padding both biases the
    per-channel magnitudes and dilutes the objective. The MoE FFN is applied
    per token (routing is hash-based on token id), so dropping positions is
    semantics-preserving -- unlike an attention input, where it would not be.
    """
    p, l = x.shape[0], x.shape[1]
    flat = x.reshape(p * l, *x.shape[2:])
    return flat[flat_idx][None]


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
    down_group_size: int | None = None,
    mask_padding: bool = False,
):
    down_group_size = down_group_size or group_size
    out = Path(out_dir)
    t0 = time.time()
    if ckpt_dir is None:
        ckpt_dir = str(out) + "-ckpt"
    print(f"[awq-dsv4] checkpointing per-layer to {ckpt_dir}", flush=True)

    print(f"[awq-dsv4] loading (lazy): {model_path}", flush=True)
    model, processor = load(model_path, lazy=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    text = _text_model(model)
    n_layers = len(text.layers)
    hc_mult = text.args.hc_mult
    sliding_window = text.args.sliding_window

    prompts = _load_filtered_prompts(dataset_file, n_prompts, set(exclude_categories))
    token_batches = _tokenize_prompts(tokenizer, prompts, max_tokens)
    print(f"[awq-dsv4] {len(token_batches)} prompts, {n_layers} layers, "
          f"bits={bits} (down={down_bits}), group_size={group_size} "
          f"(down_group_size={down_group_size})", flush=True)

    # Fixed-length right-padding: search_best_scale/search_best_clip concatenate
    # every calibration batch's captured activations into ONE tensor (axis=0),
    # which requires matching shape on every other axis -- real prompts here
    # vary in length (confirmed: a 5-token prompt vs a 384-token one), so
    # without padding the concatenate itself would fail on shape mismatch.
    #
    # The forward pass still has to run on the padded grid, but the captured
    # activations are masked back down to real tokens before AWQ sees them
    # (--no-mask-padding restores the old behaviour, which is what the
    # published v2 build was calibrated with). This matters because pad
    # positions are NOT excluded from the causal mask, so they carry real
    # activations from a degenerate repeated-pad context -- they bias
    # x.abs().mean() and dilute the mse() objective, rather than contributing
    # harmless zeros.
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

    lengths = [len(t) for t in token_batches]
    if mask_padding:
        valid_idx = _valid_index(lengths, pad_len)
        mx.eval(valid_idx)
        all_ids_awq = _drop_pad(all_ids[..., None], valid_idx)[..., 0]
        mx.eval(all_ids_awq)
        pad_pct = 100.0 * (1.0 - sum(lengths) / float(len(lengths) * pad_len))
        print(f"[awq-dsv4] padding mask ON: {sum(lengths)}/{len(lengths) * pad_len} "
              f"real tokens ({pad_pct:.1f}% padding dropped before scale search)",
              flush=True)
    else:
        valid_idx = None
        all_ids_awq = all_ids
        print("[awq-dsv4] padding mask OFF (v2-compatible calibration)", flush=True)

    for i in range(n_layers):
        layer = text.layers[i]
        sg = layer.ffn.switch_mlp

        if ckpt_dir and _load_layer_ckpt(ckpt_dir, i, sg, bits, down_bits, group_size, down_group_size):
            # Resumed from checkpoint -- skip the expensive search/clip, but
            # still need this layer's real (now AWQ-quantized) output to
            # feed the next layer's calibration input.
            new_hidden = []
            for h, mask, ids in zip(hidden, masks, input_ids_list):
                new_hidden.append(layer(h, mask, None, ids))
            mx.eval(new_hidden)
            hidden = new_hidden
            gc.collect()
            mx.clear_cache()
            print(f"[awq-dsv4] layer {i:02d}/{n_layers - 1} loaded from checkpoint, "
                  f"skipped ({time.time() - t0:.0f}s)", flush=True)
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
        #    down_proj's own capture happens in a separate replay below, on the
        #    post-masking tensors, so no catcher is installed for this pass.
        ffn_inputs = []

        new_hidden = []
        for h, mask, ids in zip(hidden, masks, input_ids_list):
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
            mx.eval(h)
            new_hidden.append(h)
        hidden = new_hidden

        raw_ffn_input = mx.concatenate(ffn_inputs, axis=0)
        mx.eval(raw_ffn_input)

        if valid_idx is not None:
            raw_ffn_input = _drop_pad(raw_ffn_input, valid_idx)
            mx.eval(raw_ffn_input)

        # down_proj's activations are captured INSIDE SwitchGLU, after it has
        # gathered/sorted x by expert -- and that layout is data dependent
        # (measured: (P, L, k, 1, inter) for tiny inputs, flat (P*L*k, 1, inter)
        # once sorting kicks in). Rather than index-map pad positions through a
        # layout that can silently permute token<->expert correspondence, just
        # replay the FFN on the exact tensors AWQ is about to search over and
        # let SwitchGLU produce whatever layout it wants.
        awq_catch = _SwitchCatcher(sg.down_proj)
        sg.down_proj = awq_catch
        mx.eval(layer.ffn(raw_ffn_input, all_ids_awq))
        sg.down_proj = awq_catch.module
        sg.down_proj.input_feat = awq_catch.input_feat
        sg.down_proj.indices = awq_catch.indices
        mx.eval(sg.down_proj.input_feat, sg.down_proj.indices)

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
            layer_kwargs={"input_ids": all_ids_awq},
            quantize_func=qf,
            n_grid=n_grid,
        )
        apply_scale(layer.ffn_norm, [sg.gate_proj, sg.up_proj], scales_gu)

        # 3. AWQ scale search: up_proj -> down_proj. down_proj's own capture
        #    (via the leaf-level _SwitchCatcher above) IS in the right
        #    layout here -- we're calling down_proj directly (block=None),
        #    not replaying through the whole ffn, so no mismatch.
        qf_down = _quantize_func(down_bits, down_group_size)
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
        sg.down_proj.weight = search_best_clip(sg.down_proj, qf_down, down_group_size, n_grid)

        # 5. Final RTN quantize with the AWQ-calibrated scale/clip applied.
        gate_q = sg.gate_proj.to_quantized(group_size=group_size, bits=bits)
        up_q = sg.up_proj.to_quantized(group_size=group_size, bits=bits)
        down_q = sg.down_proj.to_quantized(group_size=down_group_size, bits=down_bits)
        sg.gate_proj, sg.up_proj, sg.down_proj = gate_q, up_q, down_q
        mx.eval(sg.gate_proj, sg.up_proj, sg.down_proj)

        del raw_ffn_input, ffn_inputs, scales_gu, scales_d
        del gate_q, up_q, down_q
        for h in hidden:
            mx.eval(h)

        if ckpt_dir:
            _save_layer_ckpt(ckpt_dir, i, sg, bits, down_bits, down_group_size)

        gc.collect()
        mx.clear_cache()
        print(f"[awq-dsv4] layer {i:02d}/{n_layers - 1} done ({time.time() - t0:.0f}s)", flush=True)

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
    for i in range(n_layers):
        quant_cfg[f"{prefix}.layers.{i}.ffn.switch_mlp.gate_proj"] = {"bits": bits, "group_size": group_size, "mode": "affine"}
        quant_cfg[f"{prefix}.layers.{i}.ffn.switch_mlp.up_proj"] = {"bits": bits, "group_size": group_size, "mode": "affine"}
        quant_cfg[f"{prefix}.layers.{i}.ffn.switch_mlp.down_proj"] = {"bits": down_bits, "group_size": down_group_size, "mode": "affine"}

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
    ap.add_argument("--down-group-size", type=int, default=None,
                    help="group_size for down_proj only. Defaults to --group-size. "
                         "Smaller (e.g. 64) reduces down_proj's quantization error "
                         "at a real storage cost (~2.9GB for the full 43-layer "
                         "checkpoint going 128->64), independent of --down-bits.")
    ap.add_argument("--mask-padding", dest="mask_padding", action="store_true", default=False,
                    help="drop pad positions from captured activations before the AWQ "
                         "scale/clip search. MEASURED WORSE (~14%% higher real-token "
                         "reconstruction error, see docs FINDINGS S16) -- off by default; "
                         "kept because the plumbing is reusable and the negative result "
                         "is worth being able to re-run")
    ap.add_argument("--no-mask-padding", dest="mask_padding", action="store_false",
                    help="default: the published v2 calibration (padding included)")
    ap.add_argument("--n-grid", type=int, default=20)
    ap.add_argument("--exclude-categories", nargs="*", default=["multimodal"])
    ap.add_argument("--ckpt-dir", default=None,
                    help="Per-layer resumable checkpoints. Defaults to <out>-ckpt. "
                         "Rerunning with the same --out/--ckpt-dir skips already-"
                         "completed layers instead of reprocessing them.")
    a = ap.parse_args()
    awq_quantize_experts(a.model, a.dataset, a.out, a.n_prompts, a.max_tokens,
                          a.bits, a.down_bits, a.group_size, a.n_grid,
                          a.exclude_categories, a.ckpt_dir, a.down_group_size)


if __name__ == "__main__":
    main()
