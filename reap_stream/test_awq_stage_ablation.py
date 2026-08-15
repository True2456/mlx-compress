"""Which AWQ stage actually earns its keep: the scale search or the clip search?

The 28.5% no-calibration disaster removed BOTH, so it never separated them.
Measurement (S16.x) showed gate/up's input -- an RMSNorm output -- has a nearly
flat per-channel profile (max/min 1.7), which is exactly the condition under
which AWQ's scale search has little to exploit. down_proj's input (post-SwiGLU)
is the opposite: top-1% of channels carry 16.1% of energy.

So this ablates each stage separately, for each projection family, scored on
real-token FFN reconstruction error:

    rtn    : quantize(W)                      -- no calibration at all
    scale  : quantize(W*s)/s                  -- scale search only
    clip   : quantize(clip(W))                -- clip search only
    both   : quantize(clip(W*s))/s            -- production

Prediction if the hypothesis holds: `scale` buys little over `rtn` for gate/up
but meaningfully more for down_proj, while `clip` carries most of the value.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
from mlx_lm.quant.awq import search_best_scale, search_best_clip

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
    ap.add_argument("--down-bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--down-group-size", type=int, default=64)
    ap.add_argument("--n-grid", type=int, default=20)
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
    PL = a.max_tokens

    hidden, ids_list, masks = [], [], []
    for t in batches:
        ids = mx.array(list(t) + [pad_id] * (PL - len(t)))[None]
        h = _expand_hc(text.embed_tokens(ids), hc_mult)
        m = create_attention_mask(h[:, :, 0, :], None, window_size=sliding, return_array=True)
        mx.eval(h, m)
        hidden.append(h); ids_list.append(ids); masks.append(m)
    all_ids = mx.concatenate(ids_list, axis=0); mx.eval(all_ids)
    vi = _valid_index(lengths, PL); mx.eval(vi)
    ids_msk = _drop_pad(all_ids[..., None], vi)[..., 0]; mx.eval(ids_msk)

    for li in range(a.layer):
        layer = text.layers[li]
        hidden = [layer(h, m, None, i) for h, m, i in zip(hidden, masks, ids_list)]
        mx.eval(hidden)

    layer = text.layers[a.layer]
    sg = layer.ffn.switch_mlp
    sg.gate_proj = _dequant_to_switch_linear(sg.gate_proj)
    sg.up_proj = _dequant_to_switch_linear(sg.up_proj)
    sg.down_proj = _dequant_to_switch_linear(sg.down_proj)

    ffn_inputs = []
    for h, m, i in zip(hidden, masks, ids_list):
        x, post, comb = layer.attn_hc(h)
        x = layer.attn(layer.attn_norm(x), mask=m, cache=None)
        h = hc_expand(x, h, post, comb)
        x, post, comb = layer.ffn_hc(h)
        f = layer.ffn_norm(x); mx.eval(f); ffn_inputs.append(f)
    raw_pad = mx.concatenate(ffn_inputs, axis=0); mx.eval(raw_pad)
    raw_msk = _drop_pad(raw_pad, vi); mx.eval(raw_msk)

    qf = _quantize_func(a.bits, a.group_size)
    qf_d = _quantize_func(a.down_bits, a.down_group_size)
    W0 = {n: getattr(sg, n).weight for n in ("gate_proj", "up_proj", "down_proj")}
    ref = layer.ffn(raw_msk, ids_msk); mx.eval(ref)

    def restore():
        for n, w in W0.items():
            getattr(sg, n).weight = w
        mx.eval(*[getattr(sg, n).weight for n in W0])

    def err():
        o = layer.ffn(raw_msk, ids_msk); mx.eval(o)
        d = (ref.astype(mx.float32) - o.astype(mx.float32))
        return float(mx.mean(d * d).item())

    def capture_down(x, kw):
        c = _SwitchCatcher(sg.down_proj); sg.down_proj = c
        mx.eval(layer.ffn(x, kw))
        sg.down_proj = c.module
        sg.down_proj.input_feat = c.input_feat
        sg.down_proj.indices = c.indices
        mx.eval(sg.down_proj.input_feat, sg.down_proj.indices)

    # --- production-order calibration, on production (padded) inputs ---
    sg.gate_proj.input_feat = raw_pad
    sg.up_proj.input_feat = raw_pad
    capture_down(raw_pad, all_ids)
    s_gu = search_best_scale(layers=[sg.gate_proj, sg.up_proj], block=layer.ffn,
                             layer_kwargs={"input_ids": all_ids},
                             quantize_func=qf, n_grid=a.n_grid)
    mx.eval(s_gu)
    s_d = search_best_scale(layers=[sg.down_proj], block=None,
                            layer_kwargs={"indices": sg.down_proj.indices},
                            quantize_func=qf_d, n_grid=a.n_grid)
    mx.eval(s_d)
    print(f"[abl] scales ready ({time.time()-t0:.0f}s); "
          f"gate/up spread {float((s_gu.max()/s_gu.min()).item()):.2f}, "
          f"down spread {float((s_d.max()/s_d.min()).item()):.2f}", flush=True)

    def clip_of(name, w, quant, gs, feat=None):
        """search_best_clip reads module.input_feat AND module.weight. Production
        passes scaled weights (W*s) with the UNSCALED captured input, because
        apply_scale runs between capture and clip. `feat` lets us hand it the
        x/s the scaled weights will actually see at inference."""
        m = getattr(sg, name)
        saved_w, saved_f = m.weight, m.input_feat
        m.weight = w
        if feat is not None:
            m.input_feat = feat
        out = search_best_clip(m, quant, gs, a.n_grid)
        m.weight, m.input_feat = saved_w, saved_f
        mx.eval(out)
        return out

    results = []

    def run_family(tag, names, scales, quant, gs):
        for mode in ("rtn", "scale", "clip", "both", "both_fix"):
            restore()
            for n in names:
                m = getattr(sg, n)
                if mode == "rtn":
                    m.weight = quant(W0[n])
                elif mode == "scale":
                    m.weight = quant(W0[n] * scales) / scales
                elif mode == "clip":
                    m.weight = quant(clip_of(n, W0[n], quant, gs))
                elif mode == "both":
                    m.weight = quant(clip_of(n, W0[n] * scales, quant, gs)) / scales
                else:  # both_fix: clip search sees the scaled-domain input
                    m.weight = quant(
                        clip_of(n, W0[n] * scales, quant, gs,
                                feat=m.input_feat / scales)) / scales
            mx.eval(*[getattr(sg, n).weight for n in names])
            e = err()
            results.append((tag, mode, e))
            print(f"[abl] {tag:9s} {mode:6s} real-token err {e:.6e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        restore()

    run_family("gate/up", ("gate_proj", "up_proj"), s_gu, qf, a.group_size)
    # down_proj's clip search needs its captured (post-gather) features
    capture_down(raw_pad, all_ids)
    run_family("down", ("down_proj",), s_d, qf_d, a.down_group_size)

    print(f"\n=== AWQ stage ablation, layer {a.layer} (real-token FFN error) ===")
    print(f"{'family':>9} {'rtn':>13} {'scale':>13} {'clip':>13} {'both':>13} {'both_fix':>13}")
    for tag in ("gate/up", "down"):
        row = {m: e for t, m, e in results if t == tag}
        print(f"{tag:>9} " + " ".join(f"{row[m]:13.6e}" for m in ("rtn", "scale", "clip", "both", "both_fix")))
    print(f"\n{'family':>9} {'scale vs rtn':>14} {'clip vs rtn':>13} {'both vs rtn':>13} {'both_fix vs rtn':>16}")
    for tag in ("gate/up", "down"):
        row = {m: e for t, m, e in results if t == tag}
        r = row["rtn"]
        print(f"{tag:>9} " + " ".join(
            f"{100*(r-row[m])/r:12.1f}%" for m in ("scale", "clip", "both", "both_fix")))
    print(f"[abl] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
