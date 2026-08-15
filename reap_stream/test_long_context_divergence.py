"""Does quantization error on Ling-3.0-flash grow with context position?

*** CAUTION: this script forces the whole test sequence through KDA's
*** recurrent kernel as ONE giant single chunk with no cache. That is NOT
*** how real serving processes long context (which always chunks a long
*** prompt and carries recurrent/KV state between chunks). On Ling-3.0-flash
*** this single-chunk methodology produced a false catastrophic-divergence
*** result past ~13.5K tokens that completely disappeared once retested
*** with realistic chunked prefill -- see
*** reap_stream/test_chunked_prefill_divergence.py and
*** docs/LING3-LONG-CONTEXT-QUANT-FINDINGS.md. Use the chunked version to
*** judge whether a checkpoint is actually safe at long context; this
*** script is kept for other diagnostic uses (e.g. as a cheap first look,
*** or paired with localize_divergence.py) but do not trust a positive
*** result from it alone without confirming under chunked prefill too.

Motivation: DWQ calibration prompts are short (384-1024 tokens), but real
agentic sessions accumulate 96K-120K tokens of context. DWQ only trains the
quantized affine *scales* against per-token teacher logits -- it doesn't
touch routing or the KDA recurrent state directly -- so a scale correction
learned at short context plausibly still holds at long context. But this
model's hybrid attention is already known to be context-length-sensitive
(MTP acceptance/throughput degraded sharply past ~16K context in prior
testing, tied to KDA ArraysCache state). This script checks the premise
directly: build one long, realistic (tool-output-shaped) document, run it
through both the BF16 teacher (streamed, windowed) and the existing
quantized checkpoint (resident, no DWQ recovery applied), and compare
teacher-vs-quantized KL divergence + top-1 agreement at positions spread
across the sequence. If divergence grows with position, short-context-only
DWQ calibration is leaving the long-context regime under-corrected. If it
stays flat, the concern doesn't apply here.

Usage:
    .venv/bin/python -m reap_stream.test_long_context_divergence \
        --teacher models/Ling-3.0-flash \
        --quantized artifacts/ling3-8fixed-5routed \
        --target-tokens 16000 --n-buckets 10 --layers-at-once 3
"""
from __future__ import annotations

import argparse
import gc
import glob
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load

from .collect_ling3 import _free_layer, _run_layer, _text_model, load_lazy


def _build_long_document(repo_root: Path, target_tokens_estimate: int) -> str:
    """Concatenate real repo files as <file> blocks (shaped like agentic tool
    output), then a task instruction at the end. ~4 chars/token estimate to
    size the source material; the caller does the real trim by token count."""
    candidates = sorted(repo_root.glob("reap_stream/*.py")) + \
                 sorted(repo_root.glob("scripts/*.py")) + \
                 sorted(repo_root.glob("docs/*.md"))
    target_chars = target_tokens_estimate * 5  # overshoot; we trim by tokens later
    parts = []
    total = 0
    for f in candidates:
        if total >= target_chars:
            break
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        parts.append(f"<file path=\"{f.relative_to(repo_root)}\">\n{text}\n</file>")
        total += len(text)
    parts.append(
        "\n\nTASK:\nGiven the files above, summarize what this codebase does "
        "and identify the single riskiest untested assumption in it."
    )
    return "\n\n".join(parts)


def _bucket_positions(seq_len: int, n_buckets: int, lo_frac: float = 0.0,
                      hi_frac: float = 1.0) -> list[int]:
    # Evenly spaced positions within [lo_frac, hi_frac] of the sequence,
    # skipping the very first few tokens (embedding noise) by default.
    lo = max(8, int(lo_frac * seq_len))
    hi = min(seq_len - 2, int(hi_frac * seq_len))
    if n_buckets <= 1:
        return [hi]
    step = (hi - lo) / (n_buckets - 1)
    return sorted({int(lo + i * step) for i in range(n_buckets)})


def _topk_np(logits_row, k=128):
    idx = mx.argpartition(-logits_row, k)[:k]
    vals = mx.take_along_axis(logits_row, idx, axis=-1)
    order = mx.argsort(-vals)
    return mx.take_along_axis(vals, order, axis=-1), mx.take_along_axis(idx, order, axis=-1)


def _kl_from_full(teacher_row, student_row, k=128):
    """KL(teacher || student) over the teacher's own top-k, like DWQ's loss."""
    t_vals, t_idx = _topk_np(teacher_row.astype(mx.float32), k)
    s_sel = mx.take_along_axis(student_row.astype(mx.float32), t_idx, axis=-1)
    log_p_t = t_vals - mx.logsumexp(t_vals)
    log_p_s = s_sel - mx.logsumexp(s_sel)
    p_t = mx.exp(log_p_t)
    kl = (p_t * (log_p_t - log_p_s)).sum()
    top1_match = bool(mx.argmax(teacher_row).item() == mx.argmax(student_row).item())
    return float(kl.item()), top1_match


def run(teacher_path: str, quantized_path: str, target_tokens: int,
        n_buckets: int, layers_at_once: int, max_layers: int = 0,
        debug_skip_quant: bool = False, lo_frac: float = 0.0, hi_frac: float = 1.0):
    repo_root = Path(__file__).resolve().parents[1]

    print("[divergence] tokenizing document...", flush=True)
    _, tokenizer = load(quantized_path, lazy=True)  # cheap: just need the tokenizer
    doc = _build_long_document(repo_root, target_tokens)
    tokens = tokenizer.encode(doc)
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    tokens = tokens[:target_tokens]
    seq_len = len(tokens)
    positions = _bucket_positions(seq_len, n_buckets, lo_frac, hi_frac)
    print(f"[divergence] seq_len={seq_len} tokens, buckets at positions={positions}",
          flush=True)
    inp = mx.array(tokens)[None]

    # ---- teacher: streamed, windowed over all 42 layers ----
    t0 = time.time()
    print(f"[divergence] loading teacher (lazy, streamed): {teacher_path}", flush=True)
    model, _ = load_lazy(teacher_path)
    text = _text_model(model)
    n_layers = len(text.layers)
    if max_layers:
        n_layers = min(n_layers, max_layers)
        print(f"[divergence] DEBUG: truncating teacher to first {n_layers} layers "
              f"-- output is NOT a real comparison, pipeline check only", flush=True)
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
        act = mx.get_active_memory() / 1e9
        print(f"[divergence] teacher blocks {w0}-{window[-1]}/{n_layers-1} "
              f"active={act:.1f} GB ({time.time()-t0:.0f}s elapsed)", flush=True)
    h = text.norm(h)
    idx_arr = mx.array(positions)
    h_sel = mx.take(h[0], idx_arr, axis=0)               # [n_buckets, hidden]
    teacher_logits = model.lm_head(h_sel)                  # [n_buckets, vocab]
    mx.eval(teacher_logits)
    print(f"[divergence] teacher logits ready ({time.time()-t0:.0f}s)", flush=True)
    del model
    gc.collect()
    mx.clear_cache()

    # ---- quantized (existing checkpoint, no DWQ recovery): resident ----
    t1 = time.time()
    if debug_skip_quant:
        print("[divergence] DEBUG: skipping quantized load, using teacher logits "
              "+ noise as a fake student -- pipeline check only", flush=True)
        q_logits = teacher_logits + mx.random.normal(teacher_logits.shape) * 0.5
        mx.eval(q_logits)
    else:
        print(f"[divergence] loading quantized student (resident): {quantized_path}", flush=True)
        from .bailing_swiglu_clamp import apply_bailing_swiglu_clamp
        apply_bailing_swiglu_clamp()
        q_model, _ = load(quantized_path, lazy=False)
        q_logits_full = q_model(inp)                            # [1, seq, vocab]
        mx.eval(q_logits_full)
        q_logits = mx.take(q_logits_full[0], idx_arr, axis=0)  # [n_buckets, vocab]
        mx.eval(q_logits)
        del q_model, q_logits_full
        gc.collect()
        mx.clear_cache()
    print(f"[divergence] quantized logits ready ({time.time()-t1:.0f}s)", flush=True)

    # ---- compare ----
    print("\n[divergence] position | frac_of_seq | KL(teacher||quant) | top1_match")
    results = []
    for i, pos in enumerate(positions):
        kl, match = _kl_from_full(teacher_logits[i], q_logits[i])
        frac = pos / seq_len
        results.append((pos, frac, kl, match))
        print(f"  {pos:>7d} | {frac:>10.2%} | {kl:>18.4f} | {match}")

    early = [r[2] for r in results if r[1] < 0.34]
    mid = [r[2] for r in results if 0.34 <= r[1] < 0.67]
    late = [r[2] for r in results if r[1] >= 0.67]
    def avg(xs): return sum(xs) / len(xs) if xs else float("nan")
    print(f"\n[divergence] mean KL: early={avg(early):.4f} mid={avg(mid):.4f} "
          f"late={avg(late):.4f}")
    print(f"[divergence] total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--quantized", required=True)
    ap.add_argument("--target-tokens", type=int, default=16000)
    ap.add_argument("--n-buckets", type=int, default=10)
    ap.add_argument("--layers-at-once", type=int, default=3)
    ap.add_argument("--max-layers", type=int, default=0,
                    help="DEBUG: truncate teacher to first N layers, invalidates results")
    ap.add_argument("--debug-skip-quant", action="store_true",
                    help="DEBUG: skip loading the quantized model, use noised teacher logits")
    ap.add_argument("--lo-frac", type=float, default=0.0)
    ap.add_argument("--hi-frac", type=float, default=1.0)
    a = ap.parse_args()
    run(a.teacher, a.quantized, a.target_tokens, a.n_buckets, a.layers_at_once,
        a.max_layers, a.debug_skip_quant, a.lo_frac, a.hi_frac)
