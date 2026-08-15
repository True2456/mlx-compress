# Ling-3.0-flash quantization — what we did, end to end

**2026-08-10/11.** Summary of the full session: goal, tools built, bugs found,
the one big methodology mistake and its correction, the final quantization
decision, and the real accuracy numbers backing it.

## Goal

Find the smallest quantization of Ling-3.0-flash that's still safe to run,
with real evidence behind the answer rather than a guess.

## What was built

A general-purpose streaming quantize/DWQ toolkit, ported from an earlier
Step-3.7-Flash project to Ling-3.0-flash (`bailing_hybrid` architecture,
native `mlx_lm`, KDA linear attention + MLA global attention + MoE). Public
repo: **https://github.com/True2456/streaming-dwq-mlx** (private).

Key pieces, all mirrored in this repo's `reap_stream/`:

- `collect_ling3.py` — windowed streaming forward pass (loads one window of
  decoder layers at a time, frees each before the next) so the 237GB BF16
  teacher never needs to be fully resident. Also implements REAP-style
  per-expert saliency collection.
- `quantize_ling3_mixed.py` — builds a mixed-precision quantized checkpoint
  with a configurable routed-expert bit width (everything else fixed at
  8-bit): lazy-loads the teacher, quantizes, writes shards incrementally.
- `dwq_collect_targets_ling3.py` / `dwq_train_student_ling3.py` — DWQ
  (distilled weight quantization) recovery: cache the teacher's top-k
  logits over a calibration set, then train a quantized checkpoint's affine
  scales against them via KL divergence. Built and debugged, but **not
  needed** in the end — see below.
- `test_chunked_prefill_divergence.py` — **the one that matters.**
  Teacher-vs-quantized KL divergence at sampled positions across a long
  document, using realistic chunked prefill (real `KVCache`/`ArraysCache`
  carried between chunks) rather than one giant single-chunk forward pass.
- `bailing_swiglu_clamp.py`, `kda_safe_gate_patch.py` — two real bugs found
  and fixed along the way (below).

## Two real bugs found (both real, both worth keeping, neither was the
## actual quantization answer)

### 1. Missing SwiGLU clamp
`mlx_lm`'s `bailing_hybrid.py` never implements Ling-3.0-flash's trained
per-layer SwiGLU clamp (`expert_swiglu_limit_list` /
`share_expert_swiglu_limit_list` in `config.json`, layers 34-41). Running
unclamped costs **88.41% → 71.34% HumanEval** (17pp), silently — no crash,
just wrong. This was already known/patched in the user's oMLX install
before this session; ported into `reap_stream/bailing_swiglu_clamp.py` so
the streaming tools apply it too.

### 2. Missing KDA safe-gate clamp (newly found this session)
`config.json` sets `kda_safe_gate: True, kda_lower_bound: -5.0`. The HF
reference clamps KDA's recurrent decay gate to `[-5, 0)`
(`g_log = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))`); `mlx_lm`
always uses the unclamped default formula and never reads either config
field. Real, same bug class as the SwiGLU clamp (a config-specified
Ling-specific fix silently dropped in the port). Fix in
`reap_stream/kda_safe_gate_patch.py`. **Not the cause of the divergence
below** — tested directly, made no difference — but worth keeping applied
and worth reporting upstream.

## The methodology mistake (important, cost most of the session)

Early testing (`test_long_context_divergence.py` and friends) forced the
entire test sequence through KDA's recurrent kernel as **one single giant
chunk with no cache** — convenient for a one-shot KL comparison, but not
how real serving processes long context. Under that methodology, the
existing 5-bit checkpoint appeared to track the teacher perfectly through
~12K tokens then **catastrophically diverge** by ~14-16K tokens (100% of
sampled positions broken).

Seven hypotheses about the quantization scheme were tested and eliminated
in pursuit of the cause: routed-expert bit-width (5 vs 6-bit), KDA
attention weight precision, MoE router precision, quantized-kernel-vs-
plain-matmul, the missing safe-gate clamp (above), and document-specific
content. All correctly came back null, because **none of them were ever
the actual variable** — every single-chunk test (including ones using the
"working" checkpoint's own files) failed identically. The real cause: the
test methodology itself. Retested with `test_chunked_prefill_divergence.py`
(real chunked prefill, matching actual serving), **the divergence
completely disappeared** — every bucket from 0-16,000 tokens showed
uniformly small KL (0.002-0.013). The existing checkpoint was fine all
along.

**Lesson, kept prominently in the repo**: for hybrid linear-attention /
recurrent architectures (KDA, Mamba, GLA, ...), the chunking strategy used
to test long context is not a neutral detail — a single-mega-chunk test can
manufacture an entirely fictitious catastrophic failure.

## The actual quantization sweep (the real answer)

Once testing was trustworthy, built and tested routed-expert bit widths
2 through 5 (everything else fixed at 8-bit, group_size=64), using
`test_chunked_prefill_divergence.py` against the BF16 teacher:

| Routed bits | Size | Max bucket KL (64 sampled positions, 0-16K tokens) | Catastrophic positions |
|---|---|---|---|
| 5-bit (originally shipped) | 81GB | 0.013 | 0 |
| **4-bit** | **67GB** | **0.054** | **0** |
| 3-bit | 53GB | 0.137 | 0 |
| 2-bit | 39GB | 0.521 | 2 (real breakage) |

3-bit and 4-bit both showed zero catastrophic positions under KL-divergence
testing; 2-bit was the first to show real degradation. Attempted DWQ
recovery on 2-bit to see if it could be rescued, but it hit real training
costs (~160-170s/step; ~14 hours estimated for a 300-prompt pass on this
hardware, since backprop has to traverse KDA's un-fused, kernel-less
gradient path through the full decoder) — abandoned as impractical for
this session, not because it failed.

**Given real uncertainty about how well teacher-forced KL predicts actual
task accuracy** (it's a distributional proxy, not a task benchmark), the
decision was made conservatively: ship **4-bit** rather than push to 3-bit
without real accuracy validation.

## Real benchmark validation (after the decision, confirms it)

Run via oMLX's own accuracy benchmark queue (mmlu/gsm8k/humaneval),
comparing the new 4-bit checkpoint against the original 5-bit:

| Benchmark | 4-bit (67GB) | 5-bit (81GB) | Diff |
|---|---|---|---|
| MMLU | 83.0% (166/200) | 84.0% (168/200) | −1.0pp |
| GSM8K | 95.5% (191/200) | 93.5% (187/200) | **+2.0pp** |
| HumanEval | 89.0% (146/164) | 87.8% (144/164) | **+1.2pp** |

At these sample sizes, differences of 2-4 questions are within normal
binomial noise — statistically indistinguishable from parity, with 4-bit
actually ahead on two of three. Real task accuracy confirms what the KL
test suggested: **4-bit is a safe, validated choice at ~17% smaller than
what was shipped.**

3-bit was never run through real benchmarks (only KL-tested clean) —
worth doing before trusting it as the final answer, now that the benchmark
pipeline is fast and working (~2-4 min per benchmark).

## Where things ended up

- **Deployed model**: `~/.lmstudio/models/truemod/Ling-3.0-flash-8fixed-4routed`
  (67GB, moved there from this repo's `artifacts/` — LM Studio/oMLX picks
  it up from that location).
- **Original 5-bit checkpoint**: untouched, still at
  `~/.lmstudio/models/truemod/Ling-3.0-flash-8fixed-5routed` (81GB).
- **Quantization build tool**: `reap_stream/quantize_ling3_mixed.py` in
  this repo — rerun with `--routed-bits N` for any bit width; deliberately
  does *not* copy `configuration_bailing_moe_v3.py`/
  `modeling_bailing_moe_v3.py` into the output (see script docstring —
  their presence broke tokenizer loading in oMLX once, even though the
  `.py` files themselves were untouched; turned out to be a red herring for
  *that* specific bug, but harmless/correct to leave out regardless since
  `mlx_lm`/oMLX never uses them).
- **Divergence test tool**: `reap_stream/test_chunked_prefill_divergence.py`
  — use this one for any future bit-width or recovery testing, not the
  single-chunk scripts (kept for other diagnostic uses, each carries a
  caution docstring pointing here).
- **Full technical writeup** of the methodology investigation:
  [`docs/LING3-LONG-CONTEXT-QUANT-FINDINGS.md`](LING3-LONG-CONTEXT-QUANT-FINDINGS.md).
- **Public repo** (tooling + findings, portable to future work):
  https://github.com/True2456/streaming-dwq-mlx

## Unrelated but real: oMLX tokenizer compatibility bug

Found and fixed while debugging why the new checkpoint wouldn't load:
oMLX's bundled `transformers` (5.12.1) renamed `tokenizer.get_added_tokens_decoder()`
(a method, what bundled `mlx_lm` 0.31.3 calls) to `tokenizer.added_tokens_decoder`
(a property) — the old method no longer exists anywhere in the library.
This broke loading for **every** Ling checkpoint (the 5-bit one too, not
just the new one) once something updated the bundled `transformers`
version. Fixed with a one-line compat shim, installed as
`/Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_lm_added_tokens_decoder_compat.py`,
wired into `omlx/utils/model_loading.py`. Unrelated to the model/checkpoint
itself — a version mismatch between oMLX's own bundled libraries.

## Open items

1. Real-benchmark the 3-bit checkpoint (rebuild via
   `quantize_ling3_mixed.py --routed-bits 3`, run the same mmlu/gsm8k/humaneval
   queue) — the one gap in the evidence chain, now cheap to close.
2. DWQ recovery for 2-bit (or lower) — technically working pipeline, just
   needs a genuinely long background run (~14h+) if ever wanted.
3. Report the KDA safe-gate clamp bug upstream (same category as the
   already-reported SwiGLU clamp).
4. Multimodal wrapper (Gemma-4 handling vision/audio, relaying to Ling) —
   built as a standalone proxy script,
   `reap_stream/multimodal_wrapper.py`; a deeper "shows up as one model
   inside oMLX itself" version was scoped but not started (bigger, riskier
   change touching oMLX's core engine dispatch code).
