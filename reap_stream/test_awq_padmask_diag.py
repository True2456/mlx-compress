"""Diagnostic for the padding-mask A/B contradiction.

The A/B reported that scales chosen ON masked data score WORSE on masked data
than scales chosen on padded data. That cannot be true if search_best_scale's
objective is the same quantity the A/B scores with -- the masked run's pick is
by construction the grid argmin of that quantity.

So: sweep every grid ratio, score each one with the A/B's own scorer, and print
where each trial's pick actually landed. If the masked pick is not the argmin,
the harness is at fault (not the hypothesis), and this says exactly where.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
from mlx_lm.quant.awq import search_best_scale

from reap_stream.awq_quantize_deepseek_v4 import (
    _SwitchCatcher, _dequant_to_switch_linear, _drop_pad, _expand_hc,
    _load_filtered_prompts, _quantize_func, _text_model, _tokenize_prompts,
    _valid_index,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/true/Desktop/models/DeepSeek-V4-Flash-0731")
    ap.add_argument("--dataset", default="calib/ds4_agentic.jsonl")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-grid", type=int, default=20)
    ap.add_argument("--max-ratio", type=float, default=1.0,
                    help="AWQ hardcodes the exponent grid to [0,1); sweep wider to check "
                         "whether a statistic's optimum falls outside it")
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

    hidden, ids_list, masks = [], [], []
    for t in batches:
        ids = mx.array(list(t) + [pad_id] * (pad_len - len(t)))[None]
        h = _expand_hc(text.embed_tokens(ids), hc_mult)
        m = create_attention_mask(h[:, :, 0, :], None, window_size=sliding, return_array=True)
        mx.eval(h, m)
        hidden.append(h); ids_list.append(ids); masks.append(m)
    all_ids = mx.concatenate(ids_list, axis=0); mx.eval(all_ids)
    vi = _valid_index(lengths, pad_len); mx.eval(vi)
    ids_msk = _drop_pad(all_ids[..., None], vi)[..., 0]; mx.eval(ids_msk)

    for li in range(a.layer + 1):
        layer = text.layers[li]
        sg = layer.ffn.switch_mlp
        if li < a.layer:
            hidden = [layer(h, m, i) if False else layer(h, m, None, i)
                      for h, m, i in zip(hidden, masks, ids_list)]
            mx.eval(hidden)
            continue

        sg.gate_proj = _dequant_to_switch_linear(sg.gate_proj)
        sg.up_proj = _dequant_to_switch_linear(sg.up_proj)
        sg.down_proj = _dequant_to_switch_linear(sg.down_proj)

        ffn_inputs = []
        for h, m, i in zip(hidden, masks, ids_list):
            res = h
            x, post, comb = layer.attn_hc(h)
            x = layer.attn(layer.attn_norm(x), mask=m, cache=None)
            h = hc_expand(x, res, post, comb)
            x, post, comb = layer.ffn_hc(h)
            fi = layer.ffn_norm(x); mx.eval(fi); ffn_inputs.append(fi)
        raw_pad = mx.concatenate(ffn_inputs, axis=0); mx.eval(raw_pad)
        raw_msk = _drop_pad(raw_pad, vi); mx.eval(raw_msk)

    qf = _quantize_func(a.bits, a.group_size)
    W0 = {n: getattr(sg, n).weight for n in ("gate_proj", "up_proj")}
    ref = layer.ffn(raw_msk, ids_msk); mx.eval(ref)

    def score(scales):
        for n in ("gate_proj", "up_proj"):
            m = getattr(sg, n)
            m.weight = qf(W0[n] * scales) / scales
        mx.eval(sg.gate_proj.weight, sg.up_proj.weight)
        o = layer.ffn(raw_msk, ids_msk); mx.eval(o)
        d = (ref.astype(mx.float32) - o.astype(mx.float32))
        e = float(mx.mean(d * d).item())
        for n in ("gate_proj", "up_proj"):
            getattr(sg, n).weight = W0[n]
        mx.eval(sg.gate_proj.weight, sg.up_proj.weight)
        return e

    def grid_scales(x, ratio):
        x_max = x.abs().mean(axis=(0, 1))
        s = mx.maximum(x_max ** ratio, 1e-4).reshape(-1)
        return s / (s.max() * s.min()).sqrt()

    print(f"[diag] layer {a.layer}: sweeping {a.n_grid} ratios, scored on REAL tokens")
    print(f"{'ratio':>7} {'err(x_max=padded)':>19} {'err(x_max=masked)':>19}")
    best = {"padded": (None, 1e9), "masked": (None, 1e9)}
    for g in range(a.n_grid):
        r = a.max_ratio * g / a.n_grid
        ep = score(grid_scales(raw_pad, r))
        em = score(grid_scales(raw_msk, r))
        for tag, v in (("padded", ep), ("masked", em)):
            if v < best[tag][1]:
                best[tag] = (r, v)
        print(f"{r:7.2f} {ep:19.6e} {em:19.6e}", flush=True)

    print(f"\n[diag] best x_max=padded : ratio {best['padded'][0]:.2f} -> {best['padded'][1]:.6e}")
    print(f"[diag] best x_max=masked : ratio {best['masked'][0]:.2f} -> {best['masked'][1]:.6e}")

    # What did search_best_scale actually return for each?
    for tag, x, kw in (("padded", raw_pad, all_ids), ("masked", raw_msk, ids_msk)):
        sg.gate_proj.input_feat = x; sg.up_proj.input_feat = x
        c = _SwitchCatcher(sg.down_proj); sg.down_proj = c
        mx.eval(layer.ffn(x, kw))
        sg.down_proj = c.module
        sg.down_proj.input_feat = c.input_feat; sg.down_proj.indices = c.indices
        s = search_best_scale(layers=[sg.gate_proj, sg.up_proj], block=layer.ffn,
                              layer_kwargs={"input_ids": kw}, quantize_func=qf,
                              n_grid=a.n_grid)
        mx.eval(s)
        picked = score(s)
        # which grid ratio does it correspond to?
        sims = [(r_i / a.n_grid,
                 float(mx.sum(s * grid_scales(x, r_i / a.n_grid)).item()
                       / (mx.linalg.norm(s) * mx.linalg.norm(grid_scales(x, r_i / a.n_grid))).item()))
                for r_i in range(a.n_grid)]
        r_match = max(sims, key=lambda z: z[1])
        print(f"[diag] search_best_scale({tag}) -> real-token err {picked:.6e}; "
              f"closest grid ratio {r_match[0]:.2f} (cos {r_match[1]:.6f})")
    print(f"[diag] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
