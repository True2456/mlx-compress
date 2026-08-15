# Ling-3.0-flash: quantization at long context — findings (corrected)

**2026-08-10.** Investigation triggered by: does DWQ's short calibration
prompts (384-1024 tokens) miss real degradation at the long context (96K-120K
tokens) this model actually runs at in production?

**Corrected headline: no. The existing quantized checkpoint
(`ling3-8fixed-5routed`, 5-bit routed experts / 8-bit everything else) tracks
the BF16 teacher closely across the entire tested range (500-16,000 tokens)
under realistic serving conditions.** An earlier version of this document
concluded there was a catastrophic quantization failure past ~13.5K tokens.
That conclusion was wrong — it was an artifact of the test methodology, not
a real property of the model or the quantization. Both the mistake and the
seven-hypothesis elimination process that led to finding it are documented
below, because the elimination work surfaced two real, independent bugs
worth keeping.

## The actual result (chunked prefill — the correct methodology)

`reap_stream/test_chunked_prefill_divergence.py`: real `KVCache`/`ArraysCache`
carried between 1024-token chunks, matching how actual serving processes a
long prompt. Random-sampled positions across 8 buckets spanning a 16K-token
document:

| Position range | mean KL |
|---|---|
| 0 – 2,000 | 0.002 |
| 2,000 – 4,000 | 0.006 |
| 4,000 – 6,000 | 0.005 |
| 6,000 – 8,000 | 0.013 |
| 8,000 – 10,000 | 0.003 |
| 10,000 – 12,000 | 0.002 |
| 12,000 – 14,000 | 0.009 |
| 14,000 – 16,000 | 0.003 |

Zero catastrophic positions (KL > 1.0) anywhere. Uniformly small KL across
the whole range — no degradation with position at all.

## The mistake, and how it was found

Every earlier test in this investigation (`test_long_context_divergence.py`,
`localize_divergence.py`, `divergence_rate_by_length.py`) forced the entire
test sequence through KDA's recurrent kernel as **one single giant chunk,
with no cache** (`cache=None` throughout, the whole prompt in one forward
call). That's convenient for a one-shot KL comparison, but it is not how
real inference works — real serving always chunks a long prompt and carries
the recurrent/KV state between chunks via a real cache object. Under that
single-mega-chunk methodology, the same checkpoint showed:

| Position range | mean KL (single-chunk, WRONG methodology) |
|---|---|
| 500 – 12,000 | ~0.005–0.010 (looked fine) |
| 12,000 – 14,000 | 0.6 (onset) |
| 14,000 – 16,000 | **4.4 (100% catastrophic)** |

This was investigated exhaustively before the methodology itself was
questioned. Seven hypotheses were tested and eliminated, in order:

1. **Testing-harness bug** (per-layer `mx.eval`, mask recomputation vs
   reuse) — ruled out; the streaming collector was verified bit-identical
   to a direct forward once a real bug in the *diagnostic script* (see
   below) was fixed.
2. **Routed-expert bit-width** (5-bit vs 6-bit) — rebuilt the checkpoint at
   6-bit; identical failure, same onset, same magnitude to 3 decimal places.
3. **KDA attention weight precision** — rebuilt with KDA's own q/k/v/b/o
   projections left BF16 (they were already 8-bit, this went further);
   identical failure.
4. **MoE router weight precision** — rebuilt with the router left BF16;
   identical failure.
5. **Quantized kernel vs. plain matmul** — monkeypatched the quantized
   SwitchGLU path to dequantize weights and use plain `gather_mm` instead
   of the special `gather_qmm` kernel (same lossy weight *values*, different
   kernel); identical failure. Ruled out a kernel-dispatch bug.
6. **Missing KDA safe-gate clamp** — found and fixed a real, independent
   bug (see below); applying the fix did not change the failure.
7. **Document-specific content** — reran with a completely different
   document (different files, shuffled order); identical pattern. Ruled
   out a coincidental near-tie in one specific document.

None of them explained it, because none of them were the actual variable.
The real cause was that every one of these tests — including the "clean"
teacher-only comparisons — forced KDA through an unrealistically large
single chunk that real serving never produces. Retesting hypothesis-free
with `test_chunked_prefill_divergence.py`'s realistic chunking made the
"failure" disappear entirely.

**Lesson, generalizable beyond this model:** when testing a hybrid
linear-attention / recurrent architecture, the *chunking strategy* used to
process a long test sequence is not a neutral implementation detail — it
can produce a completely fictitious failure mode on its own. Always
validate a long-context test methodology against realistic chunked/cached
inference before trusting its results, especially for models with KDA,
Mamba, GLA, or similar recurrent-state mechanisms. This exact model's own
`ArraysCache`/conv-state handling was already known to be chunking-sensitive
from prior MTP work (chunked vs. stepped `ShortConv1d` processing differed
by ~2400x in numerical drift) — that should have been the first thing
checked, not the last.

## Two real bugs found along the way (independent of the above mistake)

### 1. Missing SwiGLU clamp — real, fixed, keep

`mlx_lm`'s `bailing_hybrid.py` never implements Ling-3.0-flash's trained
per-layer SwiGLU clamp (`expert_swiglu_limit_list` /
`share_expert_swiglu_limit_list` in `config.json`, layers 34-41). Measured
cost of running unclamped: 88.41% -> 71.34% HumanEval (17pp), silent, no
crash. Fix: `reap_stream/bailing_swiglu_clamp.py`. This is real, independent
of the chunking mistake, and should stay applied.

### 2. Missing KDA safe-gate clamp — real, fixed, but not the cause of the above

Ling-3.0-flash's `config.json` sets `kda_safe_gate: True,
kda_lower_bound: -5.0`. The HF reference implementation threads these into
`chunk_kda`/`fused_recurrent_kda` (`fla.ops.kda`), which use them to clamp
the recurrent decay gate's log-value to `[-5, 0)`:

```
g_log = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))     # clamped, correct
g_log = -exp(A_log) * softplus(a + dt_bias)                    # unclamped, what mlx_lm does
```

`mlx_lm.models.bailing_hybrid.KimiDeltaAttention` always uses the unclamped
form and never reads `kda_safe_gate`/`kda_lower_bound` at all, despite
`ModelArgs` declaring both fields. This is a real, config-specified,
silently-dropped piece of Ling-specific logic — same bug class as the
SwiGLU clamp, same "config field declared but never used" pattern. Fix:
`reap_stream/kda_safe_gate_patch.py`.

Tested directly (under the flawed single-chunk methodology, before the
chunking mistake was found): applying this patch to both sides of the
comparison did **not** change the catastrophic-looking result (still ~0.6 /
~4.4 mean KL at the same buckets). It has not yet been retested under
realistic chunked prefill to see whether it matters there. Worth keeping
applied regardless, since it's a real correctness issue independent of
whatever else is going on, but its practical impact (if any) under real
serving conditions is currently unverified.

### 3. A real testing-methodology bug, useful beyond this investigation

Found while trying to validate the (ultimately fictitious) single-chunk
divergence: a **quantized `nn.Linear` (`QuantizedLinear`) can silently
return wrong values for some rows of a small batch**, while the same rows
computed as part of a larger batch are correct. Verified directly: selecting
3 rows before `lm_head` (quantized) vs. projecting the full sequence then
selecting the same 3 rows afterward gave identical results for one probe
position and wildly different (KL 3-6) results for two others. An
*unquantized* `nn.Linear` does not have this problem (verified, 0.0 max
diff either way). **Always project the full batch through a quantized
layer, then select rows — never select first.**

## Tools

- `reap_stream/test_chunked_prefill_divergence.py` — **the one to use.**
  Realistic chunked-prefill teacher-vs-quantized divergence test.
- `reap_stream/test_long_context_divergence.py`,
  `reap_stream/localize_divergence.py`,
  `reap_stream/divergence_rate_by_length.py` — single-chunk methodology,
  produced the false-positive result above. Kept for other diagnostic uses
  (each carries a caution docstring); do not use alone to judge long-context
  safety.
- `reap_stream/bailing_swiglu_clamp.py` — real fix, keep applied.
- `reap_stream/kda_safe_gate_patch.py` — real fix, keep applied, impact
  under realistic serving unverified.
- `reap_stream/quantize_ling3_mixed.py` — builds mixed-precision Ling-3.0
  quantizations with a custom predicate (used for hypotheses 2-4 above).
