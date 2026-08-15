"""AWQ-calibrated quantization for Qwen3.5/3.6 (qwen3_5) checkpoints.

Scope: the dense MLP (gate_proj/up_proj/down_proj), which is 61.6% of
Qwen3.8-27B's weights and the only part quantized aggressively enough for
calibration to matter. Everything else (GatedDeltaNet, full attention, embed,
lm_head, vision) is RTN-quantized at higher bit widths chosen from the measured
imatrix -- at 5-8 bits RTN is close to lossless, and AWQ's win is a low-bit
phenomenon (measured on DeepSeek: uncalibrated 2-bit collapses to 28.5%, i.e.
chance, while calibrated scores 64.5%).

SEQUENTIAL, not parallel: layer i is calibrated on activations produced by the
already-quantized layers above it, so accumulated error is accounted for. Each
layer is visited exactly once -- the attention/GDN half of the block is
computed before quantization and reused after, since quantizing the MLP cannot
change it. That makes the whole pass ~64 layer-forwards instead of the ~2080 a
naive re-forward-per-layer would cost.

The block forward is replicated by hand (masks, rope, gdn_sink) rather than
calling the model's __call__, because we need to split the block at the MLP
boundary. `--verify-forward` asserts that replication is BIT-EXACT against
Qwen3_5DecoderLayer.__call__ before any quantization happens; it is on by
default and cheap. This is the one place a silent divergence could produce a
plausible-looking but wrong model.

Recipe defaults come from `artifacts/imatrix_qwen38.npz` concentration
analysis (participation ratio): attn q/k/v is 135x more concentrated than
down_proj, inverting DeepSeek's pattern where down_proj wanted the extra bits.
NOTE: 3-bit is deliberately never used -- oMLX's qwen35_prefill kernels expose
q2/q4/q5/q6/q8 but NOT q3 (verified at runtime), so 3-bit anywhere in the MLP
would silently drop prefill to the slow path.

Usage:
    PYTHONPATH=/Applications/oMLX.app/Contents/Resources .venv/bin/python \
        -m reap_stream.awq_quantize_qwen35 \
        --model ~/Desktop/models/Qwen3.8-27B \
        --dataset calib/qwen38_calib.jsonl \
        --out ~/Desktop/models/Qwen3.8-27B-awq \
        --n-prompts 256 --max-tokens 1024
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
from mlx_lm.quant.awq import apply_scale, search_best_clip, search_best_scale

# ---------------------------------------------------------------- recipe ----
# (bits, group_size) per tensor family. 3-bit is unavailable (no native kernel).
DEFAULT_RECIPE = {
    "mlp.gate_proj":  (4, 128),   # PR 0.0396, 41.1% of weights  -- AWQ
    "mlp.up_proj":    (4, 128),   # same input as gate_proj      -- AWQ
    "mlp.down_proj":  (4, 128),   # PR 0.2986, flattest tensor   -- AWQ
    "linear_attn.in_proj":  (5, 64),   # PR 0.0040, 14.6%
    "linear_attn.out_proj": (4, 64),   # PR 0.1188
    "self_attn.q_proj": (8, 64),  # PR 0.0022 -- most concentrated by 135x
    "self_attn.k_proj": (8, 64),
    "self_attn.v_proj": (8, 64),
    "self_attn.o_proj": (4, 64),  # PR 0.2053
    "embed_tokens":  (4, 128),
    "lm_head":       (6, 128),
    # NOTE: the MODULE path is `vision_tower`, not the checkpoint tensor name
    # `visual` -- keying on the latter silently left all 111 vision modules at
    # bf16. Both are listed so either naming works.
    "vision_tower":  (8, 128),    # NOT imatrix-measured (text-only calib)
    "visual":        (8, 128),
}


def _recipe_for(name: str, recipe: dict):
    """Longest-suffix match so `layers.3.mlp.gate_proj` hits `mlp.gate_proj`."""
    best = None
    for key, val in recipe.items():
        if key in name and (best is None or len(key) > len(best[0])):
            best = (key, val)
    return best[1] if best else None


def _qdq(bits: int, group_size: int):
    def f(w):
        q = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(*q, bits=bits, group_size=group_size)
    return f


class _Catcher(nn.Module):
    """Records a leaf Linear's input, matching mlx_lm.quant.awq's contract."""

    def __init__(self, inner):
        super().__init__()
        self.module = inner

    def __call__(self, x, *a, **k):
        if hasattr(self, "input_feat"):
            self.input_feat = mx.concatenate([self.input_feat, x], axis=0)
        else:
            self.input_feat = x
        return self.module(x, *a, **k)


# --------------------------------------------------------------- helpers ----
def _text_stack(model):
    lm = getattr(model, "language_model", None) or model
    return lm, getattr(lm, "model", lm)


def _masks(qwen_mod, h):
    """Replicate Qwen3_5Model.__call__'s mask construction for cache=None."""
    from mlx_vlm.models.qwen3_5.language import (
        _create_qwen3_5_attention_mask,
        _create_qwen3_5_ssm_mask,
    )
    n = len(qwen_mod.layers)
    cache = [None] * n
    fa = _create_qwen3_5_attention_mask(h, cache[qwen_mod.fa_idx])
    ssm = _create_qwen3_5_ssm_mask(h, cache[qwen_mod.ssm_idx])
    return fa, ssm


def _rope(qwen_mod, h, position_ids):
    """Replicate Qwen3_5Model.__call__'s position_embeddings construction.

    Only the first non-linear layer's rotary_emb is consulted, and only when it
    is not fused -- exactly as the model does it.
    """
    if position_ids is None:
        return None
    for layer in qwen_mod.layers:
        if not layer.is_linear:
            if not layer.self_attn.rotary_emb.fused_apply:
                return layer.self_attn.rotary_emb(h, position_ids)
            return None
    return None


def _attn_half(layer, x, mask, position_ids, position_embeddings):
    """First half of Qwen3_5DecoderLayer.__call__ -- unaffected by MLP quant.

    position_ids MUST be supplied: Qwen3_5Attention dereferences `cache.offset`
    unguarded when position_ids is None, so `cache=None` is not a valid calling
    convention for full-attention layers (caught by --verify-forward).
    """
    if layer.is_linear:
        r = layer.linear_attn(layer.input_layernorm(x), mask, None,
                              gdn_sink=None, target_verify=False)
    else:
        r = layer.self_attn(layer.input_layernorm(x), mask=mask, cache=None,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            target_verify=False)
    return x + r


def _verify_forward(layer, x, mask, position_ids, position_embeddings) -> float:
    """Assert the hand-split block equals the model's own block, bit-for-bit."""
    ref = layer(x, mask=mask, cache=None, position_ids=position_ids,
                position_embeddings=position_embeddings, gdn_sink=None,
                target_verify=False)
    h = _attn_half(layer, x, mask, position_ids, position_embeddings)
    ours = h + layer.mlp(layer.post_attention_layernorm(h))
    mx.eval(ref, ours)
    return float(mx.max(mx.abs(ref.astype(mx.float32) - ours.astype(mx.float32))).item())


def _load_prompts(path: str, n: int, tok, max_tokens: int):
    toks = []
    for line in open(path):
        if not line.strip():
            continue
        rec = json.loads(line)
        t = rec.get("text", "")
        if not t.strip():
            continue
        ids = tok.encode(t)[:max_tokens]
        if len(ids) > 8:
            toks.append(ids)
        if len(toks) >= n:
            break
    return toks


# ------------------------------------------------------------------ main ----
def quantize(model_path: str, dataset: str, out_dir: str, n_prompts: int,
             max_tokens: int, n_grid: int, recipe: dict, ckpt_dir: str,
             verify: bool, awq_layers: int | None) -> None:
    t0 = time.time()
    out = Path(out_dir).expanduser()
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    print(f"[awq-q35] loading {model_path}", flush=True)
    model, processor = load(model_path, lazy=False)
    tok = getattr(processor, "tokenizer", processor)
    lm, text = _text_stack(model)
    layers = text.layers
    n_layers = len(layers)
    print(f"[awq-q35] {n_layers} layers, active={mx.get_active_memory()/1e9:.1f}GB "
          f"({time.time()-t0:.0f}s)", flush=True)

    batches = _load_prompts(dataset, n_prompts, tok, max_tokens)
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0) or 0
    PL = max_tokens
    real = sum(len(b) for b in batches)
    print(f"[awq-q35] {len(batches)} prompts, seq={PL}, "
          f"{real}/{len(batches)*PL} real tokens "
          f"({100*(1-real/(len(batches)*PL)):.1f}% padding)", flush=True)

    ids = mx.array([list(b) + [pad_id] * (PL - len(b)) for b in batches])
    hidden = text.embed_tokens(ids)
    mx.eval(hidden)
    fa_mask, ssm_mask = _masks(text, hidden)
    position_ids, _ = lm.get_rope_index(input_ids=ids)
    mx.eval(position_ids)
    position_embeddings = _rope(text, hidden, position_ids)
    if position_embeddings is not None:
        mx.eval(position_embeddings)
    print(f"[awq-q35] embedded, position_ids={tuple(position_ids.shape)}, "
          f"rope={'fused' if position_embeddings is None else 'explicit'}, "
          f"active={mx.get_active_memory()/1e9:.1f}GB", flush=True)

    if verify:
        # verify on BOTH a GDN layer and a full-attention layer -- they take
        # different paths and a linear-only check would have missed the
        # cache.offset requirement entirely.
        for probe in (0, next((j for j, l in enumerate(layers) if not l.is_linear), 0)):
            lp = layers[probe]
            m = ssm_mask if lp.is_linear else fa_mask
            pid = position_ids[..., :2, :] if position_ids.ndim == 3 else position_ids[:2]
            pe = None if position_embeddings is None else tuple(
                e[:2] for e in position_embeddings)
            d = _verify_forward(lp, hidden[:2], m, pid, pe)
            kind = "GDN" if lp.is_linear else "full-attn"
            print(f"[awq-q35] layer {probe} ({kind}) replication diff {d}", flush=True)
            if d != 0.0:
                break
        if d != 0.0:
            raise RuntimeError(
                f"hand-split block forward diverges from Qwen3_5DecoderLayer "
                f"(max abs diff {d:.3e}); refusing to calibrate against a "
                f"forward pass that is not the model's own")
        print(f"[awq-q35] forward replication verified BIT-EXACT (diff {d})", flush=True)

    last = n_layers if awq_layers is None else min(awq_layers, n_layers)
    for i in range(n_layers):
        layer = layers[i]
        mask = ssm_mask if layer.is_linear else fa_mask
        mlp = layer.mlp

        # 1. attention/GDN half -- computed once, reused after MLP quantization
        h_mid = _attn_half(layer, hidden, mask, position_ids, position_embeddings)
        mx.eval(h_mid)
        mlp_in = layer.post_attention_layernorm(h_mid)
        mx.eval(mlp_in)

        if i < last:
            gb, gg = _recipe_for("mlp.gate_proj", recipe)
            db, dg = _recipe_for("mlp.down_proj", recipe)
            qf_gu, qf_d = _qdq(gb, gg), _qdq(db, dg)

            # 2. scale search: post_attention_layernorm -> {gate,up}
            mlp.gate_proj.input_feat = mlp_in
            mlp.up_proj.input_feat = mlp_in
            s_gu = search_best_scale(layers=[mlp.gate_proj, mlp.up_proj],
                                     block=mlp, layer_kwargs={},
                                     quantize_func=qf_gu, n_grid=n_grid)
            mx.eval(s_gu)
            if not bool(mx.all(mx.isfinite(s_gu)).item()):
                raise RuntimeError(f"layer {i}: non-finite gate/up scales")
            apply_scale(layer.post_attention_layernorm,
                        [mlp.gate_proj, mlp.up_proj], s_gu)

            # 3. scale search: up_proj -> down_proj (capture the SwiGLU output)
            catch = _Catcher(mlp.down_proj)
            mlp.down_proj = catch
            mx.eval(mlp(mlp_in))
            mlp.down_proj = catch.module
            mlp.down_proj.input_feat = catch.input_feat
            mx.eval(mlp.down_proj.input_feat)
            s_d = search_best_scale(layers=[mlp.down_proj], block=None,
                                    layer_kwargs={}, quantize_func=qf_d,
                                    n_grid=n_grid)
            mx.eval(s_d)
            if not bool(mx.all(mx.isfinite(s_d)).item()):
                raise RuntimeError(f"layer {i}: non-finite down_proj scales")
            apply_scale(mlp.up_proj, [mlp.down_proj], s_d)

            # 4. clip search (production order: after apply_scale, as measured
            #    correct on DeepSeek -- the "domain-corrected" variant was worse)
            mlp.gate_proj.weight = search_best_clip(mlp.gate_proj, qf_gu, gg, n_grid)
            mlp.up_proj.weight = search_best_clip(mlp.up_proj, qf_gu, gg, n_grid)
            mlp.down_proj.weight = search_best_clip(mlp.down_proj, qf_d, dg, n_grid)

            for m_ in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                if hasattr(m_, "input_feat"):
                    del m_.input_feat

            # 5. real quantization
            mlp.gate_proj = mlp.gate_proj.to_quantized(group_size=gg, bits=gb)
            mlp.up_proj = mlp.up_proj.to_quantized(group_size=gg, bits=gb)
            mlp.down_proj = mlp.down_proj.to_quantized(group_size=dg, bits=db)
            mx.eval(mlp.gate_proj, mlp.up_proj, mlp.down_proj)

        # 6. advance the stream through the NOW-QUANTIZED mlp (sequential AWQ)
        hidden = h_mid + mlp(mlp_in)
        mx.eval(hidden)
        del h_mid, mlp_in
        gc.collect()
        mx.clear_cache()
        if i % 4 == 0 or i == n_layers - 1:
            print(f"[awq-q35] layer {i:02d}/{n_layers-1} "
                  f"active={mx.get_active_memory()/1e9:.1f}GB "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ----- everything outside the MLP: RTN at the recipe's bit widths --------
    # nn.quantize walks the module tree itself and calls the predicate with the
    # dotted path; returning a dict sets per-module bits/group_size. Hand-rolled
    # traversal via vars() silently finds NOTHING on MLX modules (they are dict
    # subclasses) -- it reported "0 quantized, 0 skipped" rather than failing.
    print(f"[awq-q35] RTN-quantizing non-MLP modules ({time.time()-t0:.0f}s)",
          flush=True)
    picked, skipped = {}, []

    def _pred(path: str, mod: nn.Module):
        if isinstance(mod, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return False                      # already AWQ-quantized MLP
        if not hasattr(mod, "to_quantized"):
            return False
        if "pos_embed" in path:
            skipped.append(f"{path}(pos-embed-kept)")
            return False
        spec = _recipe_for(path, recipe)
        if spec is None:
            skipped.append(f"{path}(no-recipe)")
            return False
        b, g = spec
        w = getattr(mod, "weight", None)
        if w is None or w.shape[-1] % g != 0:
            skipped.append(f"{path}(dim{tuple(w.shape) if w is not None else '?'}%{g})")
            return False
        picked[f"{b}b/gs{g}"] = picked.get(f"{b}b/gs{g}", 0) + 1
        return {"bits": b, "group_size": g}

    nn.quantize(model, class_predicate=_pred)
    mx.eval(model.parameters())
    print(f"[awq-q35] RTN quantized: {picked}", flush=True)
    if skipped:
        from collections import Counter
        kinds = Counter(s_.split("(")[-1] for s_ in skipped)
        print(f"[awq-q35] left unquantized: {len(skipped)}  {dict(kinds)}", flush=True)
        print("   e.g. " + ", ".join(skipped[:5]), flush=True)

    # ------------------------------------------------------------- save -----
    from mlx_vlm.utils import save_weights, save_config
    out.mkdir(parents=True, exist_ok=True)
    save_weights(out, model)
    src = Path(model_path).expanduser()
    cfg = json.loads((src / "config.json").read_text())
    qcfg = {"group_size": 128, "bits": 4, "mode": "affine"}
    for name, module in _named_quantized(model):
        qcfg[name] = {"group_size": module.group_size, "bits": module.bits}
    cfg["quantization"] = qcfg
    save_config(cfg, out / "config.json")
    for f in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
              "video_preprocessor_config.json", "chat_template.jinja",
              "vocab.json", "merges.txt", "generation_config.json"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    sz = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    print(f"[awq-q35] saved -> {out}  {sz/1e9:.2f}GB  ({time.time()-t0:.0f}s)",
          flush=True)


def _named_quantized(model):
    """(path, module) for every quantized leaf -- needed so config.json records
    the per-tensor scheme. Omitting it makes the model silently load at the
    wrong bit width (hit exactly this on the DeepSeek oMLX conversion)."""
    from mlx.utils import tree_flatten
    out = []
    def walk(mod, prefix=""):
        for name, child in mod.children().items():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, list):
                for j, c in enumerate(child):
                    if isinstance(c, nn.Module):
                        walk(c, f"{full}.{j}")
            elif isinstance(child, nn.Module):
                if isinstance(child, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
                    out.append((full, child))
                else:
                    walk(child, full)
    walk(model)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="calib/qwen38_calib.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--n-grid", type=int, default=20)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--awq-layers", type=int, default=None,
                    help="AWQ only the first N layers (smoke testing)")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the bit-exact forward-replication assertion")
    a = ap.parse_args()
    quantize(a.model, a.dataset, a.out, a.n_prompts, a.max_tokens, a.n_grid,
             DEFAULT_RECIPE, a.ckpt_dir or (a.out + "-ckpt"), a.verify,
             a.awq_layers)


if __name__ == "__main__":
    main()
