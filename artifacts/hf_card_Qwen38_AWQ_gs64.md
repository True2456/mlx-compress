---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - omlx
  - awq
  - quantized
  - qwen3_5
  - vision-language
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B — AWQ 5.0bpw (group size 64, multimodal calibration)

AWQ-calibrated quantization of `Qwen/Qwen3.8-27B` (27.8B dense, 64 layers,
hybrid GatedDeltaNet + full attention, 256K context, MTP head, vision tower)
for Apple Silicon via **oMLX**.

**17.36 GB · BPW 5.00 · 3.2× smaller than bf16**

A group-size-64 variant of
[Qwen3.8-27B-AWQ-4.85bpw](https://huggingface.co/True2456/Qwen3.8-27B-AWQ-4.85bpw),
calibrated with images as well as text. Costs +0.53 GB over that build and buys
two things: full prefill speed with no workaround, and a vision tower that was
actually exercised during calibration.

| component | bits | why |
|---|---|---|
| MLP `gate_proj` / `up_proj` / `down_proj` | **4-bit gs64** | 62% of weights; gs64 reaches oMLX's NAX kernel path |
| GDN `in_proj_{qkv,z,a,b}` | 5-bit gs64 | 2nd most concentrated (participation ratio 0.0040) |
| GDN `out_proj` | 4-bit gs64 | middling |
| attention `q/k/v_proj` | 8-bit gs64 | most concentrated by 135× (PR 0.0022), only 4% of weights |
| attention `o_proj` | 4-bit gs64 | flat |
| `embed_tokens` | 4-bit gs128 | lookup, not a matmul |
| `lm_head` | 6-bit gs128 | output projection over a 248k vocab |
| vision tower | 8-bit gs128 | now calibrated (see below) |
| MTP head | 8/6/4-bit | rejection-verified drafter |

No 3-bit anywhere: oMLX's `qwen35_prefill` kernels expose q2/q4/q5/q6/q8 but
not q3, so 3-bit in the MLP silently drops prefill to a slow path.

## Why group size 64

oMLX routes 4-bit MLP matmuls to a native kernel above 2048 tokens, and that
kernel's speed comes from the NAX tensor-unit path — which is gated to
`group_size == 64`. At gs128 it demotes to a slower path, costing ~1.8× prefill
on M5 hardware unless you disable the routing
([jundot/omlx#2657](https://github.com/jundot/omlx/pull/2657)).

Measured at `pp 4096 / tg 128`, no environment variables, same machine:

| build | ppTPS | TTFT |
|---|---|---|
| gs128, kernel routing active | 513.2 | 7982 ms |
| gs128 + routing disabled | 930.6 | 4401 ms |
| **this build (gs64)** | **894.4** | **4580 ms** |

So gs64 gets essentially full prefill speed **with no workaround and no local
patching**, which is the practical argument for the extra 0.53 GB.

## Speed

M5 Max, oMLX, no environment variables. Two separate options are on: **native
MTP** (the checkpoint's built-in multi-token-prediction heads, accelerating
decode) and **SpecPrefill** with a small draft model (accelerating prefill above
8K tokens).

A note on the SpecPrefill drafter, because the obvious choice is not the fast
one. SpecPrefill feeds the *target's* token ids straight to the drafter, so a
drafter from the same tokenizer family (`vocab_size` 248,320, e.g.
`Qwen/Qwen3.5-0.8B`) is the only one whose importance scores describe the actual
prompt — a smaller-vocab drafter gets zero vectors for every out-of-range id
rather than an error, and shares only 0.19% of id-to-token meanings with this
model.

Measured, though, a 4-bit Qwen2.5-0.5B drafter is **faster and much lighter** at
long context — 1467 vs 876 ppTPS and 33.1 vs 53.1 GB peak at 200K — because the
drafter must prefill every token to score it, and a Qwen3.5-family drafter
carries head_dim 256, GDN state and a 248k-row embedding.

Both figures above are throughput only. Whether the tokenizer mismatch costs
long-context *accuracy* is not measured here, and the quality benchmarks in this
card all sit below SpecPrefill's 8192-token trigger, so they do not answer it
either. Test retrieval at depth on your own workload before committing.

| context | TTFT | TPOT | ppTPS | tgTPS | E2E | peak mem |
|---|---|---|---|---|---|---|
| pp 1k / tg 128 | 1224 ms | 23.5 ms | 837 | 42.8 | 4.2 s | 17.6 GB |
| pp 4k / tg 128 | 5214 ms | 17.8 ms | 786 | 56.5 | 7.5 s | 19.2 GB |
| pp 8k / tg 128 | 10109 ms | 18.6 ms | 810 | 54.3 | 12.5 s | 20.2 GB |
| pp 16k / tg 128 | 6565 ms | 20.3 ms | 2496 | 49.6 | 9.2 s | 21.4 GB |
| pp 32k / tg 128 | 14766 ms | 21.2 ms | 2219 | 47.5 | 17.5 s | 22.8 GB |
| pp 64k / tg 128 | 32029 ms | 23.8 ms | 2046 | 42.3 | 35.1 s | 24.7 GB |
| pp 128k / tg 128 | 75917 ms | 25.1 ms | 1727 | 40.1 | 79.1 s | 28.5 GB |
| pp 200k / tg 128 | 136381 ms | 33.1 ms | 1467 | 30.4 | 140.6 s | 33.1 GB |

Two separate accelerators are active, and they show up in different columns.
**Native MTP** drives the decode side (tgTPS). **SpecPrefill** drives the
prefill side: a small draft model scores token importance and only the top ~20%
are prefilled on the target. It engages above 8192 tokens, which is exactly
where ppTPS steps from ~810 to ~2500 and TTFT *falls* from 10.1 s at 8K to
6.6 s at 16K. Those tail rows are still cold prefills — they are sparse, not
cached — so they are legitimately faster, just not measuring the same work as
the ≤8K rows.

The tail also shows the memory story: the full 200K window costs 33.1 GB and
still decodes at 30 tok/s.

Batched, same machine:

| batch | tgTPS | avg TTFT | E2E | speedup |
|---|---|---|---|---|
| 1 | 42.8 | 1224 ms | 4.2 s | 1.00× |
| 2 | 47.6 | 3253 ms | 9.4 s | 1.11× |
| 4 | 77.2 | 5315 ms | 15.3 s | 1.80× |
| 8 | 115.9 | 10158 ms | 27.2 s | 2.71× |

## Calibration

352 prompts × 1024 tokens, 234,477 real tokens, rendered through Qwen's own
chat template with a 50/50 think/nothink split. Unlike the 4.85bpw build, this
one includes images, so the vision tower actually executed during calibration
rather than having its bits assigned by guesswork:

| domain | share |
|---|---|
| text (coding, tool use, agentic, reasoning) | 66% |
| charts (ChartQA) | 11% |
| natural photographs (VQAv2) | 11% |
| GLSL / raymarching renders | 11% |

Sequential AWQ over the dense MLP: each layer is calibrated on activations from
the already-quantized layers above it. Everything outside the MLP is RTN at the
widths above.

## MTP

The MTP head ships **inside** the checkpoint (31 tensors under `mtp.*`) and is
quantized: 8-bit attention, 6-bit `fc`, 4-bit MLP. Turn it on with oMLX's
`mtp_enabled`, which uses those heads to draft and verify during decode.

> **Fixed 2026-08-16 — re-download if you pulled this before that date.**
> The head shipped in raw-HF norm convention while the backbone was already
> converted to MLX's, so `mtp.layers.0.input_layernorm` averaged 0.0361 instead
> of ~1.036 and two norms were negative. Drafts stopped matching the target and
> acceptance collapsed. On a loader with no compensation this made MTP *slower*
> than no speculation at all (22.4 vs 24.6 tok/s); repaired, the same setup runs
> 43.7 tok/s. Inside oMLX the damage was partly masked — its `norm_repair`
> shifts any head norm averaging below 0.5, which caught 3 of the 7 and left
> `q_norm` (0.78), `k_norm` (0.79) and `mtp.norm` (1.25) raw; repairing those
> three measured **+9% decode** (median 53.3 vs 48.8 tok/s, `pp 4096 / tg 128`,
> three runs each, no overlap between the groups).
>
> Only `model-00005-of-00005.safetensors` changed. Benchmarks below are
> unaffected: they were run with MTP off, and MTP is rejection-verified, so a
> degraded head costs acceptance rate but never output correctness.

Quantizing the head is safe — drafts are rejection-verified by the target model,
so head error costs acceptance rate, never output correctness. Leaving it at
bf16 inside a quantized model measurably *halves* throughput.

MTP is a distinct feature from speculative decoding with an external draft
model, and the two accelerate different phases. They compose: the speed table
above was measured with native MTP on the decode side **and** SpecPrefill
driving prefill from a Qwen2.5-0.5B-Instruct draft model, which is what produces
the ~3× prefill step above 8K.

## Quality

Same questions, same harness, against the
[4.85bpw sibling](https://huggingface.co/True2456/Qwen3.8-27B-AWQ-4.85bpw):

| benchmark | this build (5.0bpw gs64) | 4.85bpw gs128 |
|---|---|---|
| HumanEval | 91.5% (150/164) | 93.3% (153/164) |
| GSM8K | 92.5% (185/200) | 92.0% (184/200) |
| MMLU | 84.0% (168/200) | 83.0% (166/200) |
| **total** | **503 / 564** | **503 / 564** |

The totals are identical. Three questions lost on HumanEval, three regained
across GSM8K and MMLU — that is what noise looks like at this sample size, not a
capability difference, and neither per-benchmark gap is significant on its own.
Both builds sit four questions behind bf16.

So the extra 0.53 GB does not buy accuracy. It buys the prefill speed above
without an oMLX patch, and a vision tower that was calibrated on real images
rather than assigned its bit width by guesswork. Pick on that basis.

One caveat on attribution: this build changes **two variables at once** relative
to the 4.85bpw one — group size *and* calibration data. Since the aggregate
result is a tie, neither variable moved the needle enough to need separating.

## Requires oMLX

The checkpoint keeps its `mtp.*` weights so the MTP head can bind. Stock
`mlx_vlm` sees those keys, flips its `should_shift_norm_weights` heuristic, and
applies a `+1.0` offset to layernorms that already have it — the model still
loads and still produces fluent text, but the text is wrong. oMLX patches
`sanitize` to gate that on `conv1d` layout instead, which is correct here.

Do not load this outside oMLX.

## Sampling

The chat template takes `reasoning_effort` of `xhigh` (default), `medium` or
`low`, and raises on anything else. It also accepts `enable_thinking` and
`preserve_thinking`. Pass via `chat_template_kwargs`.

## Caveats

- 27 vision `linear_fc2` modules stay at bf16: their input dim (4304) is not
  divisible by any supported group size.
- Built with [mlx-compress](https://github.com/True2456/mlx-compress).
