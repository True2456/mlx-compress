"""AWQ-calibrated quantization for Ling-3.0-flash's routed experts
(mlp.switch_mlp.{gate_proj,up_proj,down_proj}) -- everything else (KDA/MLA
attention, dense MLP layers, router, shared_experts, embed, lm_head) gets
plain RTN quantization at --fixed-bits, matching the existing
quantize_ling3_mixed.py policy. Only the routed experts get the expensive
AWQ scale/clip search, mirroring awq_quantize_deepseek_v4.py's scoping
rationale but structurally simpler here:

- bailing_hybrid has no Hyper-Connections -- DecoderLayer.__call__ is a
  plain `h = x + attn(norm(x)); h + mlp(norm(h))`, so no HC expand/collapse
  bookkeeping is needed around the reference forward pass.
- Two mask kinds depending on `layer.is_global`: create_attention_mask for
  MLA (global) layers, create_ssm_mask for KDA (recurrent linear-attention)
  layers -- see collect_ling3.py's _run_layer, mirrored here.
- The teacher ships BF16 (not natively pre-quantized like DeepSeek-V4's
  fp8), so switch_mlp is already a plain-weight SwitchGLU/SwitchLinear --
  no dequantize-native-format step needed before AWQ.
- mlx_lm.quant.awq's search_best_scale/search_best_clip/apply_scale are
  reused as-is (verified generic in awq_quantize_deepseek_v4.py); no
  cross-package monkey-patch needed here since bailing_hybrid uses mlx_lm's
  own stock SwitchGLU/SwitchLinear classes directly (unlike oMLX's custom
  DeepSeek-V4 patch, which needed one).
- Ling's trained SwiGLU clamp (bailing_swiglu_clamp) and KDA safe-gate
  clamp (kda_safe_gate_patch) are applied before loading -- both are real,
  previously-found correctness bugs in mlx_lm's stock bailing_hybrid (see
  docs/LING3-QUANTIZATION-SESSION-SUMMARY.md); running calibration without
  them would capture activations from the wrong (uncorrected) forward pass.

Usage:
    .venv/bin/python -m reap_stream.awq_quantize_ling3 \
        --model models/Ling-3.0-flash \
        --dataset calib/ds4_agentic_ling3.jsonl \
        --out artifacts/ling3-8fixed-3routed-awq \
        --n-prompts 256 --max-tokens 768 --bits 3 --fixed-bits 8 --group-size 64
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
import numpy as np
from mlx_lm.utils import load, save_config, save_model
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.quant.awq import apply_scale, search_best_clip, search_best_scale

# Below this many calibration-token activations, an expert's AWQ scale/clip
# search is fit on too little data to trust -- fall back to plain RTN for
# it. See the comment at the splice site for the measured distribution and
# the real KL-spike case this fixes.
LOW_EXPERT_THRESHOLD = 32

from .bailing_swiglu_clamp import apply_bailing_swiglu_clamp
from .kda_safe_gate_patch import apply_kda_safe_gate


def _text_model(model):
    return model.model


def _is_moe_layer(layer) -> bool:
    import mlx_lm.models.bailing_hybrid as bh
    return isinstance(layer.mlp, bh.SparseMoeBlock)


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
    # calib/ds4_agentic_ling3.jsonl records are already fully rendered
    # through Ling3's own chat_template.jinja -- encode directly, no
    # re-wrapping through apply_chat_template.
    batches = []
    for p in prompts:
        tokens = tokenizer.encode(p)
        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]
        batches.append(list(tokens)[:max_tokens])
    return batches


class _SwitchCatcher(nn.Module):
    """Wraps a SwitchGLU projection to record its real input features and
    MoE routing indices, matching mlx_lm.quant.awq's Catcher contract."""

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


def awq_quantize_ling3(
    model_path: str,
    dataset_file: str,
    out_dir: str,
    n_prompts: int,
    max_tokens: int,
    bits: int,
    fixed_bits: int,
    group_size: int,
    n_grid: int,
    exclude_categories: list[str],
):
    t0 = time.time()
    apply_bailing_swiglu_clamp()
    apply_kda_safe_gate()

    print(f"[awq-ling3] loading (lazy): {model_path}", flush=True)
    model, tokenizer, config = load(model_path, lazy=True, return_config=True)
    text = _text_model(model)
    n_layers = len(text.layers)

    prompts = _load_filtered_prompts(dataset_file, n_prompts, set(exclude_categories))
    token_batches = _tokenize_prompts(tokenizer, prompts, max_tokens)
    print(f"[awq-ling3] {len(token_batches)} prompts, {n_layers} layers, "
          f"routed(switch_mlp)={bits}-bit, fixed={fixed_bits}-bit, group_size={group_size}",
          flush=True)

    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
    pad_len = max_tokens
    hidden = []
    for tokens in token_batches:
        padded = list(tokens) + [pad_id] * (pad_len - len(tokens))
        ids = mx.array(padded)[None]
        h = text.word_embeddings(ids)
        mx.eval(h)
        hidden.append(h)

    n_moe = 0
    moe_layer_ids: list[int] = []
    for i in range(n_layers):
        layer = text.layers[i]

        if layer.is_global:
            mask = create_attention_mask(hidden[0], None, return_array=True)
            gla_mask = None
        else:
            mask = None
            gla_mask = create_ssm_mask(hidden[0], None)

        if not _is_moe_layer(layer):
            new_hidden = []
            for h in hidden:
                m = mask if layer.is_global else gla_mask
                h_out = layer(h, mask=m, cache=None)
                new_hidden.append(h_out)
            mx.eval(new_hidden)
            hidden = new_hidden
            gc.collect()
            mx.clear_cache()
            print(f"[awq-ling3] layer {i:02d}/{n_layers - 1} (dense/attn-only, no AWQ) "
                  f"done ({time.time() - t0:.0f}s)", flush=True)
            continue

        n_moe += 1
        moe_layer_ids.append(i)
        sg = layer.mlp.switch_mlp

        # 1. Reference forward through attn + norm to capture the real FFN
        #    input (post_attention_layernorm(h)), same shape/content the
        #    switch_mlp actually sees inside SparseMoeBlock.__call__.
        ffn_inputs = []
        down_catch = _SwitchCatcher(sg.down_proj)
        sg.down_proj = down_catch

        new_hidden = []
        for h in hidden:
            m = mask if layer.is_global else gla_mask
            r = layer.attention(layer.input_layernorm(h), m, None)
            h2 = h + r
            ffn_in = layer.post_attention_layernorm(h2)
            mx.eval(ffn_in)
            ffn_inputs.append(ffn_in)

            mlp_out = layer.mlp(ffn_in)
            h_out = h2 + mlp_out
            mx.eval(h_out)
            new_hidden.append(h_out)
        hidden = new_hidden

        sg.down_proj = down_catch.module
        sg.down_proj.input_feat = down_catch.input_feat
        sg.down_proj.indices = down_catch.indices

        raw_ffn_input = mx.concatenate(ffn_inputs, axis=0)
        mx.eval(raw_ffn_input)

        # Per-expert calibration coverage: AWQ's scale/clip search is only as
        # good as the activation sample it's fit on. Routing is heavily
        # skewed (measured: median ~135 activations/expert but some experts
        # see <10 out of 512, in a run that otherwise averages 384) -- a
        # rarely-routed expert's AWQ scale is fit on a handful of noisy
        # samples and can end up WORSE than plain RTN, which needs no
        # calibration data at all. Confirmed directly: one such
        # under-sampled expert (10 calibration hits) was the sole cause of a
        # real KL spike (1.27, vs 0.06-0.32 everywhere else in the same
        # bucket) at a real eval position that happened to route through it.
        # Fix: quantize low-count experts with plain RTN instead of AWQ.
        expert_counts = np.bincount(
            np.array(down_catch.indices).flatten(), minlength=sg.gate_proj.weight.shape[0]
        )
        low_experts = np.where(expert_counts < LOW_EXPERT_THRESHOLD)[0]
        if len(low_experts) > 0:
            print(f"[awq-ling3] layer {i:02d}: {len(low_experts)}/{len(expert_counts)} "
                  f"experts under {LOW_EXPERT_THRESHOLD} calibration activations "
                  f"(min={expert_counts.min()}) -- falling back to RTN for those",
                  flush=True)
        # Original (pre-AWQ-scale) weights, needed for the RTN fallback --
        # apply_scale below mutates .weight in place, so this copy has to
        # happen before that.
        orig_weights = {
            "gate_proj": sg.gate_proj.weight,
            "up_proj": sg.up_proj.weight,
            "down_proj": sg.down_proj.weight,
        }
        mx.eval(*orig_weights.values())

        # 2. AWQ scale search: post_attention_layernorm -> {gate_proj, up_proj}.
        qf = _quantize_func(bits, group_size)
        sg.gate_proj.input_feat = raw_ffn_input
        sg.up_proj.input_feat = raw_ffn_input
        scales_gu = search_best_scale(
            layers=[sg.gate_proj, sg.up_proj],
            block=layer.mlp,
            layer_kwargs={},
            quantize_func=qf,
            n_grid=n_grid,
        )
        apply_scale(layer.post_attention_layernorm, [sg.gate_proj, sg.up_proj], scales_gu)

        # 3. AWQ scale search: up_proj -> down_proj.
        scales_d = search_best_scale(
            layers=[sg.down_proj],
            block=None,
            layer_kwargs={"indices": sg.down_proj.indices},
            quantize_func=qf,
            n_grid=n_grid,
        )
        apply_scale(sg.up_proj, [sg.down_proj], scales_d)

        # 4. Clip search.
        sg.gate_proj.weight = search_best_clip(sg.gate_proj, qf, group_size, n_grid)
        sg.up_proj.weight = search_best_clip(sg.up_proj, qf, group_size, n_grid)
        sg.down_proj.weight = search_best_clip(sg.down_proj, qf, group_size, n_grid)

        # 5. Final RTN quantize with the AWQ-calibrated scale/clip applied.
        gate_q = sg.gate_proj.to_quantized(group_size=group_size, bits=bits)
        up_q = sg.up_proj.to_quantized(group_size=group_size, bits=bits)
        down_q = sg.down_proj.to_quantized(group_size=group_size, bits=bits)

        if len(low_experts) > 0:
            low_mask = mx.array(np.isin(np.arange(len(expert_counts)), low_experts))
            for name, q_mod in (("gate_proj", gate_q), ("up_proj", up_q), ("down_proj", down_q)):
                plain_w, plain_s, plain_b = mx.quantize(
                    orig_weights[name], bits=bits, group_size=group_size
                )
                m_w = low_mask.reshape(-1, *([1] * (q_mod.weight.ndim - 1)))
                m_s = low_mask.reshape(-1, *([1] * (q_mod.scales.ndim - 1)))
                q_mod.weight = mx.where(m_w, plain_w, q_mod.weight)
                q_mod.scales = mx.where(m_s, plain_s, q_mod.scales)
                q_mod.biases = mx.where(m_s, plain_b, q_mod.biases)
            mx.eval(gate_q, up_q, down_q)

        sg.gate_proj, sg.up_proj, sg.down_proj = gate_q, up_q, down_q
        mx.eval(sg.gate_proj, sg.up_proj, sg.down_proj)

        del raw_ffn_input, ffn_inputs, scales_gu, scales_d
        del gate_q, up_q, down_q
        for h in hidden:
            mx.eval(h)

        gc.collect()
        mx.clear_cache()
        print(f"[awq-ling3] layer {i:02d}/{n_layers - 1} (MoE #{n_moe}) AWQ done "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"[awq-ling3] AWQ pass done, quantizing everything else uniformly at "
          f"{fixed_bits}-bit and saving -> {out_dir}", flush=True)

    def quant_predicate(path: str, module):
        if "switch_mlp" in path:
            return False  # already AWQ-quantized above; don't re-touch
        return {"group_size": group_size, "bits": fixed_bits}

    from mlx_lm.utils import quantize_model
    q_model, q_config = quantize_model(
        model, config, group_size=group_size, bits=fixed_bits,
        quant_predicate=quant_predicate,
    )

    # quant_predicate returning False for switch_mlp paths correctly skips
    # re-quantizing them (they're already AWQ-quantized above), but
    # quantize_model then never records a per-path config entry for them
    # either -- on reload, the loader falls back to the top-level
    # group_size/bits (fixed_bits, e.g. 8), constructing switch_mlp modules
    # at the wrong bit-width and failing with a weight shape mismatch
    # against the actually-saved bits-bit packed tensors. Inject the real
    # per-path overrides explicitly.
    quant_cfg = q_config.setdefault("quantization", {})
    for i in moe_layer_ids:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            quant_cfg[f"model.layers.{i}.mlp.switch_mlp.{proj}"] = {
                "bits": bits, "group_size": group_size, "mode": "affine",
            }
    q_config["quantization_config"] = quant_cfg

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    save_model(out, q_model, donate_model=True)
    save_config(q_config, config_path=out / "config.json")
    tokenizer.save_pretrained(out)
    src = Path(model_path)
    for pattern in ("generation_config.json", "chat_template.jinja"):
        for fp in src.glob(pattern):
            shutil.copy(fp, out)

    print(f"[awq-ling3] done in {time.time() - t0:.0f}s -> {out} "
          f"({n_moe} MoE layers AWQ-calibrated on {len(token_batches)} prompts from {dataset_file})",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--fixed-bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--n-grid", type=int, default=20)
    ap.add_argument("--exclude-categories", nargs="*", default=["multimodal"])
    a = ap.parse_args()
    awq_quantize_ling3(a.model, a.dataset, a.out, a.n_prompts, a.max_tokens,
                        a.bits, a.fixed_bits, a.group_size, a.n_grid,
                        a.exclude_categories)


if __name__ == "__main__":
    main()
