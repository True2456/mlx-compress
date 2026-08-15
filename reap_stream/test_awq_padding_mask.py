"""A/B: does masking pad positions out of AWQ's calibration produce better
expert scales?

The comparison has to be scored carefully. AWQ's search picks scales that
minimise mse(out, out_q) over whatever positions it was handed, so scoring the
padded run on padded data would just measure "did it fit its own objective" --
it trivially would. Both scale sets are therefore scored on the SAME
real-tokens-only reconstruction error, which is what the deployed model
actually has to do well on.

    scales_pad    = search_best_scale(padded activations)
    scales_masked = search_best_scale(real-token activations)
    error(s)      = mse( ffn_fp(x_real), ffn_quant(s)(x_real) )

Lower error = better. Run:
    PYTHONPATH=/Applications/oMLX.app/Contents/Resources .venv/bin/python \
        -m reap_stream.test_awq_padding_mask --layers 0 1 2
"""
from __future__ import annotations

import argparse
import gc
import time

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
from mlx_lm.quant.awq import search_best_scale
from mlx_lm.quant.awq import run_layer as _rl  # noqa: F401  (patched on import)

from reap_stream.awq_quantize_deepseek_v4 import (
    _SwitchCatcher,
    _dequant_to_switch_linear,
    _drop_pad,
    _expand_hc,
    _load_filtered_prompts,
    _quantize_func,
    _text_model,
    _tokenize_prompts,
    _valid_index,
)


def _mse(a, b):
    d = (a.astype(mx.float32) - b.astype(mx.float32))
    return float(mx.mean(d * d).item())


def _score(ffn, sg, scales, qf_gu, qf_d, x, ids, ref):
    """Quantize gate/up (and down via its own qf) under `scales`, return the
    real-token reconstruction error of the whole FFN block."""
    saved = {n: getattr(sg, n).weight for n in ("gate_proj", "up_proj")}
    for n in ("gate_proj", "up_proj"):
        m = getattr(sg, n)
        m.weight = qf_gu(m.weight * scales) / scales
    mx.eval(sg.gate_proj.weight, sg.up_proj.weight)
    out_q = ffn(x, ids)
    mx.eval(out_q)
    err = _mse(ref, out_q)
    for n, w in saved.items():
        getattr(sg, n).weight = w
    mx.eval(sg.gate_proj.weight, sg.up_proj.weight)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/true/Desktop/models/DeepSeek-V4-Flash-0731")
    ap.add_argument("--dataset", default="calib/ds4_agentic.jsonl")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--down-bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--down-group-size", type=int, default=64)
    ap.add_argument("--n-grid", type=int, default=20)
    ap.add_argument("--reverse", action="store_true",
                    help="run the masked trial first -- control for state leaking between trials")
    a = ap.parse_args()

    t0 = time.time()
    model, processor = load(a.model, lazy=True)
    tok = getattr(processor, "tokenizer", processor)
    text = _text_model(model)
    hc_mult, sliding = text.args.hc_mult, text.args.sliding_window

    prompts = _load_filtered_prompts(a.dataset, a.n_prompts, {"multimodal"})
    batches = _tokenize_prompts(tok, prompts, a.max_tokens)
    lengths = [len(t) for t in batches]
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0) or 0
    pad_len = a.max_tokens
    real, grid = sum(lengths), len(lengths) * pad_len
    print(f"[ab] {len(batches)} prompts, {real}/{grid} real tokens "
          f"({100*(1-real/grid):.1f}% padding)", flush=True)

    hidden, ids_list, masks = [], [], []
    for t in batches:
        ids = mx.array(list(t) + [pad_id] * (pad_len - len(t)))[None]
        h = _expand_hc(text.embed_tokens(ids), hc_mult)
        m = create_attention_mask(h[:, :, 0, :], None, window_size=sliding, return_array=True)
        mx.eval(h, m)
        hidden.append(h); ids_list.append(ids); masks.append(m)
    all_ids = mx.concatenate(ids_list, axis=0); mx.eval(all_ids)

    vi = _valid_index(lengths, pad_len); mx.eval(vi)
    ids_masked = _drop_pad(all_ids[..., None], vi)[..., 0]; mx.eval(ids_masked)

    results = []
    for li in range(max(a.layers) + 1):
        layer = text.layers[li]
        sg = layer.ffn.switch_mlp
        run_ab = li in a.layers

        if not run_ab:
            hidden = [layer(h, m, None, i) for h, m, i in zip(hidden, masks, ids_list)]
            mx.eval(hidden); gc.collect(); mx.clear_cache()
            continue

        sg.gate_proj = _dequant_to_switch_linear(sg.gate_proj)
        sg.up_proj = _dequant_to_switch_linear(sg.up_proj)
        sg.down_proj = _dequant_to_switch_linear(sg.down_proj)

        ffn_inputs, new_hidden = [], []
        for h, m, i in zip(hidden, masks, ids_list):
            res = h
            x, post, comb = layer.attn_hc(h)
            x = layer.attn(layer.attn_norm(x), mask=m, cache=None)
            h = hc_expand(x, res, post, comb)
            res = h
            x, post, comb = layer.ffn_hc(h)
            fi = layer.ffn_norm(x); mx.eval(fi); ffn_inputs.append(fi)
            x = layer.ffn(fi, i)
            h = hc_expand(x, res, post, comb); mx.eval(h); new_hidden.append(h)

        raw_pad = mx.concatenate(ffn_inputs, axis=0); mx.eval(raw_pad)
        raw_msk = _drop_pad(raw_pad, vi); mx.eval(raw_msk)

        qf = _quantize_func(a.bits, a.group_size)
        qf_d = _quantize_func(a.down_bits, a.down_group_size)

        # Reference: full-precision FFN output on REAL tokens only.
        ref = layer.ffn(raw_msk, ids_masked); mx.eval(ref)

        def _capture_down(x, ids_kw):
            """Replay the FFN so SwitchGLU produces down_proj's activations in
            its own (data-dependent, possibly expert-sorted) layout for exactly
            these tokens -- avoids index-mapping pad positions through it."""
            c = _SwitchCatcher(sg.down_proj); sg.down_proj = c
            mx.eval(layer.ffn(x, ids_kw))
            sg.down_proj = c.module
            sg.down_proj.input_feat = c.input_feat
            sg.down_proj.indices = c.indices
            mx.eval(sg.down_proj.input_feat, sg.down_proj.indices)

        trials = {}
        _order = [("padded", raw_pad, all_ids), ("masked", raw_msk, ids_masked)]
        if a.reverse:
            _order.reverse()
        for tag, rawx, kw_ids in _order:
            sg.gate_proj.input_feat = rawx
            sg.up_proj.input_feat = rawx
            _capture_down(rawx, kw_ids)
            s_ = search_best_scale(
                layers=[sg.gate_proj, sg.up_proj], block=layer.ffn,
                layer_kwargs={"input_ids": kw_ids}, quantize_func=qf, n_grid=a.n_grid,
            )
            mx.eval(s_)
            trials[tag] = (s_, _score(layer.ffn, sg, s_, qf, qf_d, raw_msk, ids_masked, ref))
            print(f"[ab] L{li} {tag:6s} real-token err {trials[tag][1]:.6e} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        ep, em = trials["padded"][1], trials["masked"][1]
        cos = float(mx.mean(
            trials["padded"][0] * trials["masked"][0]
            / (mx.linalg.norm(trials["padded"][0]) * mx.linalg.norm(trials["masked"][0]))
        ).item() * trials["padded"][0].size)
        delta = 100.0 * (ep - em) / ep
        results.append((li, ep, em, delta, cos))
        print(f"[ab] L{li} -> masked is {delta:+.2f}% better  (scale cos-sim {cos:.5f})",
              flush=True)

        hidden = new_hidden
        del raw_pad, raw_msk, ffn_inputs, ref, trials
        gc.collect(); mx.clear_cache()

    print("\n=== AWQ padding-mask A/B (real-token reconstruction error) ===")
    print(f"{'layer':>6} {'padded':>13} {'masked':>13} {'improvement':>12} {'cos':>9}")
    for li, ep, em, d, c in results:
        print(f"{li:6d} {ep:13.6e} {em:13.6e} {d:+11.2f}% {c:9.5f}")
    if results:
        avg = sum(r[3] for r in results) / len(results)
        print(f"\nmean improvement from masking: {avg:+.2f}%")


if __name__ == "__main__":
    main()
