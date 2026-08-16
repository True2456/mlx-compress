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

# Qwen3.8-27B — AWQ 4.85bpw (MLX / oMLX)

AWQ-calibrated quantization of `Qwen/Qwen3.8-27B` (27.8B dense, 64 layers,
hybrid GatedDeltaNet + full attention, 256K context, MTP head, SigLIP-style
vision tower) for Apple Silicon via **oMLX**.

**16.83 GB · BPW 4.85 · 3.3× smaller than bf16**

4-bit on the MLP, which is 62% of the weights, raised selectively elsewhere
where measurement said it mattered. That averages out to 4.85 bits per weight.
Bit widths came from an importance matrix collected on this model, not from
another model's recipe. Details below.

| component | params | bits | why |
|---|---|---|---|
| MLP `gate_proj` / `up_proj` | 11.41B | 4-bit gs128 | 41% of weights; AWQ-calibrated |
| MLP `down_proj` | 5.70B | 4-bit gs128 | flattest tensor in the model; AWQ-calibrated |
| GDN `in_proj_{qkv,z,a,b}` | 4.05B | 5-bit gs64 | 2nd most concentrated |
| GDN `out_proj` | 1.51B | 4-bit gs64 | middling |
| attention `q/k/v_proj` | 1.17B | 8-bit gs64 | most concentrated by 135× |
| attention `o_proj` | 0.50B | 4-bit gs64 | flat |
| `embed_tokens` | 1.27B | 4-bit gs128 | lookup, not a matmul |
| `lm_head` | 1.27B | 6-bit gs128 | output projection over a 248k vocab |
| vision tower | 0.46B | 8-bit gs128 | see caveat |
| MTP head | 0.43B | 8/6/4-bit | rejection-verified, see below |

Nothing is quantized to 3-bit: oMLX's `qwen35_prefill` kernels expose
q2/q4/q5/q6/q8 but not q3, so 3-bit anywhere in the MLP would silently drop
prefill to a slow path.

## Quality

Measured against the bf16 original on identical prompts, via oMLX's own
`omlx.eval` harness.

| benchmark | this build | bf16 | delta |
|---|---|---|---|
| HumanEval | 93.3% (153/164) | 93.9% (154/164) | −1 question |
| GSM8K | 92.0% (184/200) | 92.5% (185/200) | −1 question |
| MMLU | 83.0% (166/200) | 84.0% (168/200) | −2 questions |

**Four questions out of 564.** The aggregate difference (89.2% vs 89.9%) is
well inside noise for these sample sizes.

Quantizing from bf16 at ~4.8 BPW is a very different regime from squeezing an
already-4-bit checkpoint to ~2.9 BPW, where the same pipeline costs 17 MMLU
points. There is real headroom here and the recipe spends it carefully.

## Speed

M5 Max, 128 GB, oMLX, MTP enabled, `pp N / tg 128`.

| context | TTFT | TPOT | prefill tok/s | gen tok/s | peak mem |
|---|---|---|---|---|---|
| 1k | 1163 ms | 15.4 ms | 880 | 65.5 | 17.0 GB |
| 4k | 4401 ms | 19.8 ms | 931 | 50.9 | 18.5 GB |
| 8k | 10001 ms | 16.9 ms | 819 | 59.7 | 19.4 GB |
| 16k | 21411 ms | 17.1 ms | 765 | 59.0 | 21.3 GB |
| 32k | 53288 ms | 17.6 ms | 615 | 57.2 | 25.0 GB |
| 64k | 119769 ms | 24.2 ms | 547 | 41.7 | 32.6 GB |

Against the bf16 original: **3.9× generation** (55–65 vs 14 tok/s), prefill at
parity, and a third of the memory. At 64k context this build uses 32.6 GB where
bf16 needs 68.5 GB.

## MTP

Native `mtp_enabled` works and is worth turning on: **1.8–2.1× generation**,
88.7% draft acceptance, 3.05 tokens per backbone forward, with the drafter
costing about 1% of backbone time. (MTP is a distinct option from speculative
decoding with an external draft model — they accelerate different phases.)

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
> Only `model-00005-of-00005.safetensors` changed. Benchmarks in this card are
> unaffected: they were run with MTP off, and MTP is rejection-verified, so a
> degraded head costs acceptance rate but never output correctness.

The MTP head is quantized (8-bit attention, 6-bit `fc`, 4-bit MLP). This is
safe: drafts are verified by the target model through rejection sampling, so
drafter error costs acceptance rate, never output correctness. Leaving it at
bf16 inside an otherwise quantized model measurably *halves* throughput, since
it then sits outside oMLX's quantized verify path.

Draft acceptance at depth 3 was 96%, so `mtp_num_draft_tokens` above the
default of 3 may be worth trying.

## Two things to know before using this

**1. oMLX only.** The checkpoint keeps its `mtp.*` weights so the MTP head can
bind. Stock `mlx_vlm` sees those keys, flips its `should_shift_norm_weights`
heuristic, and applies a `+1.0` offset to layernorms that already have it. The
model still loads and still produces fluent text, but the text is garbage. oMLX
patches `sanitize` to gate that on `conv1d` layout instead, which is correct
here. Do not load this outside oMLX.

**2. Prefill needs an environment variable until
[jundot/omlx#2657](https://github.com/jundot/omlx/pull/2657) ships.** oMLX
routes 4-bit gs128 MLP matmuls to a native kernel above 2048 tokens, but that
kernel's speed comes from the NAX tensor-unit path, which is gated to
group_size 64. At gs128 it demotes to a slower path:

```
OMLX_QWEN35_Q4_MLP_MIN_TOKENS=999999999
OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS=999999999
```

Without it, prefill at 4k drops from 931 to 513 tok/s (TTFT 4.4s → 8.0s). The
speed table above was measured with it set.

## Sampling

The chat template takes `reasoning_effort` with values `xhigh` (default),
`medium`, or `low`, and raises on anything else. It also accepts
`enable_thinking` and `preserve_thinking`. Pass them via
`chat_template_kwargs`.

For long-context work, enable KV cache quantization.

## How it was built

Sequential AWQ over the dense MLP: each layer is calibrated on activations
produced by the already-quantized layers above it, so accumulated error is
accounted for. Everything outside the MLP is RTN at the bit widths above, where
RTN is close to lossless.

Calibration was 192 prompts × 1024 tokens of coding, tool-use and agentic
conversations, re-rendered through Qwen's own chat template with a 50/50
think/nothink split. Rendering matters: calibrating on raw text rather than
templated conversations cost 27 MMLU points on an earlier model in this line.

Bit allocation came from an importance matrix collected over the same corpus,
scored by participation ratio (what fraction of input channels actually carry
the energy):

| family | top-1% energy share | participation ratio |
|---|---|---|
| attention q/k/v | 65.0% | 0.0022 |
| GDN `in_proj` | 59.4% | 0.0040 |
| MLP gate/up | 25.0% | 0.0396 |
| `lm_head` | 23.5% | 0.0920 |
| GDN `out_proj` | 22.6% | 0.1188 |
| attention `o_proj` | 17.0% | 0.2053 |
| MLP `down_proj` | 13.8% | 0.2986 |

Attention q/k/v is 135× more concentrated than `down_proj`, so it gets 8-bit
despite being only 4% of the weights, while `down_proj` gets no premium despite
being 20%. This is the opposite of what a DeepSeek-V4 recipe would suggest,
where `down_proj` is the concentrated tensor. Transferring that intuition would
have spent bits on the family that needed them least.

## Caveats

- **Vision is not measured.** Calibration was text-only, so the tower never ran
  and its 8-bit assignment is a conservative guess rather than a measurement.
  27 `linear_fc2` modules stay at bf16 because their input dim (4304) is not
  divisible by any supported group size.
- Shard filenames are inconsistent (`00001-of-00004` through `00005-of-00005`).
  The index is authoritative and loading is unaffected.
