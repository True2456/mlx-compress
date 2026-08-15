# mlx-compress — reference

Detail behind `SKILL.md`. Source docs live in
[mlx-compress/docs](https://github.com/True2456/mlx-compress/tree/master/docs):
`QWEN38-FINDINGS.md`, `DWQ-DISTRIBUTED-FINDINGS.md`, `HANDOFF.md`.

---

## AWQ: what each stage does

Two stages, both load-bearing, and they overlap heavily. Measured on
DeepSeek-V4 layer 1, real-token FFN reconstruction error vs plain RTN:

| | scale only | clip only | both |
|---|---|---|---|
| gate/up | −57.5% | −60.8% | −64.9% |
| down_proj | −59.4% | −32.2% | −47.2% |

Two things follow. Neither stage can be dropped. And scaling's advantage over
clipping is **proportional to input concentration** — `down_proj` (concentrated)
gains far more from scaling, `gate/up` (flat) is indifferent.

Do **not** conclude from the `down_proj` row that clip should be skipped there:
that reverses at depth. At layer 40, scale-only is 1.495 vs both at 0.224 — an
85% regression the other way, on errors ~1000× larger in absolute terms.

`x_max` is a **conditioning heuristic, not an estimator**. It only has to
produce a scale vector that quantizes well; the grid search then measures real
output error. Four separate attempts to feed it a *more faithful* input all
measured worse: masking pad tokens, substituting an imatrix, and both
directions of a clip-domain correction. Do not reason about what it "should"
see.

AWQ's exponent grid is hardcoded to `[0, 1)` (`ratio = g/n_grid`). `n_grid`
sets resolution, not range. If you change the input statistic, its optimum may
fall outside that window and you will measure the grid, not the change.

**Sequential vs parallel.** Calibrate layer *i* on activations from the
already-quantized layers above it. The attention half of a block is unaffected
by MLP quantization, so compute it once and reuse it — that makes the pass
~N layer-forwards instead of ~N²/2.

---

## Architecture notes

### Qwen3.5 / 3.6 / 3.8 (`qwen3_5`)

- Dense, hybrid: `full_attention_interval: 4` → mostly GatedDeltaNet layers,
  GDN is ~20% of weights.
- **No 3-bit kernel.** oMLX's `qwen35_prefill` exposes q2/q4/q5/q6/q8 only.
  3-bit anywhere in the MLP silently drops prefill to a slow path, so the
  compression ladder jumps 4-bit → 2-bit with nothing between.
- Attention dereferences `cache.offset` when `position_ids is None`, so
  `cache=None` is not a valid calling convention. Get position ids from the
  model's own `get_rope_index` rather than reimplementing mrope.
- `get_input_embeddings` returns `InputEmbeddingsFeatures`, not an array.
- For image rows, embed at **full length first**; truncating `input_ids` before
  embedding desynchronises image placeholders from `pixel_values`.
- mrope returns `(3, batch, seq)` for image rows and `(batch, seq)` for text.
  Normalise before batching.

### DeepSeek-V4 (`deepseek_v4`)

- Hyper-Connections: hidden state is `(batch, seq, hc_mult, hidden)`.
- Hash-routed MoE needs `input_ids` threaded through every block.
- Scope AWQ to `ffn.switch_mlp`; whole-block quantization would touch
  attention/compressor/indexer that should stay at native precision.
- `mlx_lm.quant.awq`'s isinstance checks reference `mlx_lm`'s `SwitchLinear`;
  an `mlx_vlm` model has a structurally identical but *different* class, and the
  mismatch silently no-ops the perturbation step rather than erroring. Patch the
  module-level name.
- SwitchGLU's captured activation layout is **data dependent** — `(P, L, k, 1,
  inter)` for tiny inputs, expert-sorted flat at real sizes. Never index-map
  into it; replay the forward on the tensors you want captured.

### GDN / KDA hybrids (Ling-3.0 `bailing_hybrid`, Qwen3.5)

- Per-layer cache is `ArraysCache` for linear layers, `KVCache` for full
  attention. A missing handler override silently killed prefix caching once.
- NAX (M5 tensor units) qmm is **group_size 64 only**. At gs128 it demotes to a
  plain Metal path measured 3.6× slower than MLX's own qmm on M5 — while on
  pre-NAX M3 the same kernel *wins* at both group sizes. Hardware-dependent;
  measure before assuming. (jundot/omlx#2657)

---

## REAP adapter surface

Five hooks make the streaming loop architecture-agnostic. Reference
implementation: `reap_streaming/adapters/mlx_lm_moe.py`.

| hook | contract |
|---|---|
| `text_model(model)` | unwrap the VLM/causal-LM shell to the thing with `.layers` |
| `moe_layer_ids(text)` | which layer indices are MoE (dense layers still run, for hidden-state carry) |
| `num_experts(text)` | routed experts per MoE layer, assumed uniform |
| `embed(text, ids)` | initial hidden state, including any architecture-specific scaling |
| `install_probe(text, i, stats)` | mutate layer *i* in place so it accumulates (expert_ids, gate_weights, activation_norms) without changing its output |
| `run_layer(layer, h)` | one block forward, handling this architecture's mask/cache conventions; **must `mx.eval()` before returning** |
| `free_layer(text, i)` | drop the layer's weights |

The same surface drives imatrix collection — only the probe differs.

## Distributed (two machines)

Only worth it when the backward pass will not fit. Quantized-MoE backward needs
~26 GB per layer (dequantized `[n_experts, out, in]` weights plus gradients),
stacking linearly and nearly independent of sequence length.

- `mx.distributed.send` has **no VJP rule**. Treat the boundary hidden state as
  a differentiable argument and send the gradient back as a seed cotangent.
  Validate bit-exact against a single-process reference.
- Convention used here: rank 0 owns the **last** layers and the head, so
  `position = world_size - rank - 1`.
- `mx.checkpoint` does not help — it elides forward activations, but this memory
  is allocated inside the backward. Measured byte-identical with and without.
  It also has a silent failure mode: **closure-captured parameters produce a
  correct loss with all-zero gradients**. Thread parameters as explicit
  arguments and abort if the gradient norm is zero.
- Merge shards with APFS clone + in-place byte patching, not a full rewrite
  (5 GB of writes instead of 101 GB).
- Link-local addresses change. Re-derive them; do not hardcode.

---

## Negative results — do not re-run these

| Attempt | Result |
|---|---|
| DWQ to recover quantization damage | KL −89–91%, MMLU p=0.65, P(EOS) at turn starts 0.0004 → 0.0795 |
| Feed AWQ an imatrix | No gain. AWQ already computes a per-channel activation statistic and then measures real output error |
| Mask pad tokens from AWQ calibration | Worse at every layer (+17.5% / +7.0% / +4.4% error) |
| Give the clip search its true scaled-domain input | Worse on both families |
| oQ/oQe as the quantizer | 44.0% MMLU at 118.4 GB vs AWQ 64.5% at 108.2 GB — though confounded (no imatrix, borrowed sensitivity, MTP stripped), so a floor rather than a verdict |
| REAM (merge instead of prune) | PPL improved 0.194 but accuracy was flat (24/24 prune vs 23/24 REAM). PPL over-credits merging |
| Per-layer bit allocation from concentration | Only 3 of 43 layers qualified; ~1 GB. Not worth the build |
| Per-expert bit allocation | Blocked: runtime gathers experts into one stacked tensor requiring uniform bits |

---

## Sizing arithmetic

Effective bits per weight include group metadata: `bits + 32/group_size` for
fp16 scale + fp16 bias. So gs128 adds 0.25, gs64 adds 0.50. Halving group size
costs the same as adding a quarter-bit, and buys finer granularity plus, on some
hardware, a faster kernel path.

Rough anchors: a dense 27B at 4.85 BPW is 16.8 GB and loses ~4 benchmark
questions of 564 vs bf16. A 304B MoE at 2.91 BPW is 108 GB and loses ~17 MMLU
points vs its (already 4-bit) parent.
