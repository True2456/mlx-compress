"""Is teacher-vs-quantized catastrophic divergence a LENGTH effect, or just
ordinary quantization noise hitting this model's known near-tied-logit
positions (documented independently in Documents/Ling-3.0-flash-omlx-findings.md
sec 2.5, unrelated to quantization)?

*** CAUTION: like test_long_context_divergence.py, this forces the whole
*** sequence through KDA's recurrent kernel as one single chunk with no
*** cache -- not representative of real serving. It correctly established
*** that the divergence IS a length effect (not a near-tie coincidence),
*** but the divergence itself turned out to be a single-chunk testing
*** artifact -- see reap_stream/test_chunked_prefill_divergence.py and
*** docs/LING3-LONG-CONTEXT-QUANT-FINDINGS.md. The rate-by-length technique
*** here is still valid and reusable; just run it (or something like it)
*** under chunked prefill, not single-chunk, to draw conclusions about real
*** deployment safety.

localize_divergence.py showed hidden-state error grows gradually and
similarly for a pre-cliff and post-cliff probe through most of the network
(cos_sim never drops below ~0.989) -- the catastrophic KL only appears after
lm_head amplifies a modest perturbation into an argmax flip. That's
consistent with "quantization noise occasionally flips a near-tie," which
could happen at ANY position, not preferentially at long context.

This script tests that directly: sample many random positions in a SHORT
window and a LONG window of the SAME document, compute KL at each, and
compare the RATE of catastrophic divergence (KL > threshold) between the two
regimes. If the rate is similar, it's not a length effect -- it's ordinary
quantization noise interacting with tie-proneness that exists everywhere. If
the long-context rate is much higher, that supports genuine length-dependent
degradation.

Usage:
    .venv/bin/python -m reap_stream.divergence_rate_by_length \
        --teacher models/Ling-3.0-flash --quantized artifacts/ling3-8fixed-5routed \
        --target-tokens 16000 --n-samples 25 --short-hi 2000 --long-lo 10000
"""
from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load

from .collect_ling3 import _free_layer, _run_layer, _text_model, load_lazy
from .test_long_context_divergence import _build_long_document, _kl_from_full


def run(teacher_path: str, quantized_path: str, target_tokens: int, n_samples: int,
        short_hi: int, long_lo: int, kl_threshold: float, layers_at_once: int, seed: int):
    rng = random.Random(seed)
    repo_root = Path(__file__).resolve().parents[1]

    print("[rate] tokenizing document...", flush=True)
    _, tok = load(quantized_path, lazy=True)
    doc = _build_long_document(repo_root, target_tokens)
    tokens = tok.encode(doc)
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    tokens = tokens[:target_tokens]
    seq_len = len(tokens)
    inp = mx.array(tokens)[None]

    short_positions = sorted(rng.sample(range(8, short_hi), n_samples))
    long_positions = sorted(rng.sample(range(long_lo, seq_len - 2), n_samples))
    all_positions = short_positions + long_positions
    idx_arr = mx.array(all_positions)
    print(f"[rate] seq_len={seq_len}, {n_samples} short positions in [8,{short_hi}), "
          f"{n_samples} long positions in [{long_lo},{seq_len}), threshold={kl_threshold}",
          flush=True)

    print(f"[rate] loading teacher (streamed): {teacher_path}", flush=True)
    model, _ = load_lazy(teacher_path)
    text = _text_model(model)
    n_layers = len(text.layers)
    h = text.word_embeddings(inp)
    mx.eval(h)
    for w0 in range(0, n_layers, layers_at_once):
        window = range(w0, min(w0 + layers_at_once, n_layers))
        for li in window:
            h = _run_layer(text.layers[li], h)
        for li in window:
            _free_layer(text, li)
        gc.collect()
        mx.clear_cache()
    h = text.norm(h)
    h_sel = mx.take(h[0], idx_arr, axis=0)
    teacher_logits = model.lm_head(h_sel)     # unquantized lm_head: small-batch is safe (verified)
    mx.eval(teacher_logits)
    del model
    gc.collect()
    mx.clear_cache()

    print(f"[rate] loading quantized (resident): {quantized_path}", flush=True)
    from .bailing_swiglu_clamp import apply_bailing_swiglu_clamp
    apply_bailing_swiglu_clamp()
    q_model, _ = load(quantized_path, lazy=False)
    q_logits_full = q_model(inp)              # full-sequence: quantized lm_head, must not slice first (verified)
    mx.eval(q_logits_full)
    q_logits = mx.take(q_logits_full[0], idx_arr, axis=0)
    mx.eval(q_logits)
    del q_model, q_logits_full
    gc.collect()
    mx.clear_cache()

    def summarize(label, positions, offset):
        kls = []
        n_bad = 0
        for i, pos in enumerate(positions):
            kl, match = _kl_from_full(teacher_logits[offset + i], q_logits[offset + i])
            kls.append(kl)
            if kl > kl_threshold:
                n_bad += 1
        kls_sorted = sorted(kls)
        n = len(kls_sorted)
        median = kls_sorted[n // 2]
        mean = sum(kls) / n
        print(f"[rate] {label}: n={n} mean_kl={mean:.4f} median_kl={median:.4f} "
              f"max_kl={max(kls):.4f} rate(KL>{kl_threshold})={n_bad}/{n} "
              f"({100*n_bad/n:.1f}%)")
        return kls

    print()
    short_kls = summarize("SHORT", short_positions, 0)
    long_kls = summarize("LONG ", long_positions, n_samples)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--quantized", required=True)
    ap.add_argument("--target-tokens", type=int, default=16000)
    ap.add_argument("--n-samples", type=int, default=25)
    ap.add_argument("--short-hi", type=int, default=2000)
    ap.add_argument("--long-lo", type=int, default=10000)
    ap.add_argument("--kl-threshold", type=float, default=1.0)
    ap.add_argument("--layers-at-once", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.teacher, a.quantized, a.target_tokens, a.n_samples, a.short_hi, a.long_lo,
        a.kl_threshold, a.layers_at_once, a.seed)
