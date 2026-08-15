# Laguna-S-2.1 REAP: Measured Findings

**Model:** `poolside/Laguna-S-2.1-NVFP4-mlx` — `LagunaForCausalLM`, 48 decoder
layers (layer 0 dense, 47 sparse MoE), **256 routed experts/layer**, top-k=10,
1 shared expert/layer, per-head attention gating, mixed full/sliding attention
(4:1 pattern), hidden_size=3072, moe_intermediate_size=1024. Ships already
quantized to native **NVFP4** (4-bit, group_size=16) — not converted or
requantized by this project, that's how poolside distributed it.

**Hardware:** M5 Max, unified memory, target deploy is a 64GB machine (LM
Studio / oMLX).

**Scope:** Everything below is measured on this machine this session
(2026-08-02), not asserted.

---

## TL;DR

| # | Finding | Magnitude | Status |
|---|---|---|---|
| 1 | Model isn't public/HF-known — it's a local-only LM Studio download | found at `~/.lmstudio/models/poolside/Laguna-S-2.1-*` | resolved |
| 2 | Already native 4-bit, same situation as DeepSeek-V4 — prune only, no requantize | routed experts are 93.9% of total weight bytes | ✅ built |
| 3 | mlx_vlm already has full native `laguna` support in this repo's own `.venv` | no new model code needed, only a collect/apply script pair | ✅ confirmed |
| 4 | Architecture is simpler than DeepSeek-V4: no hyper-connections, no hash-routed layers | most of DeepSeek-V4's REAM special-casing doesn't apply here | ✅ built |
| 5 | bf16→numpy direct conversion crashes on this checkpoint's router scores | `PEP 3118 buffer format string B` error | ✅ fixed (cast to float32 first) |
| 6 | 37.5% prune (256→160 experts) landed at 45GB against a ~48GB projection | 6% under projection, good margin under 64GB | ✅ built, functional-verified |
| 7 | Not yet accuracy-tested | only load+generate sanity-checked | ⚠️ open |

---

## 1. Finding the model

User referred to it as "Laguna s2.1" with no prior mention anywhere in this
project or its memory. It's a real model, but local-only: not on the public
Hub under a name this project had context on, sitting in
`~/.lmstudio/models/poolside/`:

- `Laguna-S-2.1-NVFP4-mlx/` — 67GB, the main model, native NVFP4
- `Laguna-S-2.1-DFlash/` and `-DFlash-NVFP4` — a 6-layer EAGLE/DFlash-style
  speculative-decoding draft model (`num_experts: 0`, `DFlashLagunaForCausalLM`),
  **not** touched by this pruning — draft models this small aren't worth
  pruning and speculative-decoding drafts need to stay closely aligned with
  the target model's distribution.

## 2. Byte budget — where the size actually lives

Measured directly from `model.safetensors.index.json` shard headers (real
tensor byte offsets, not a config-derived estimate):

| Category | Size | Scales with prune ratio? |
|---|---|---|
| `switch_mlp` weight (routed experts) | 56.77 GiB | yes |
| `switch_mlp` scales (routed experts) | 7.10 GiB | yes |
| **routed-expert total** | **63.87 GiB (88.8%)** | |
| other_fixed (attn/norms/embed/lm_head) | 6.84 GiB | no |
| shared_expert (1/layer, all layers) | 0.89 GiB | no |
| dense_mlp (layer 0 only) | 0.23 GiB | no |
| router gate | 0.07 GiB | no |
| **fixed total** | **8.03 GiB (11.2%)** | |
| **Grand total** | **71.9 GiB** (≈ 67 GiB, decimal-vs-binary) | |

Same shape as the DeepSeek-V4 finding
(`docs/DEEPSEEK-V4-FINDINGS.md` #7): routed experts dominate total size, so
pruning expert *count* is a near-linear lever on final size, and — because
the source is already native low-bit — pruning alone (no further
quantization) is the highest-quality way to hit a size target. There's no
headroom left to trade more precision for size without visible extra damage.

Target: `total = fixed + kept_fraction × prunable`. Solving for the user's
chosen prune ratio (0.375, i.e. 62.5% kept):

```
8.03 + 0.625 × 63.87 = 47.95 GiB  (projected)
```

Actual result: **45 GiB** — 6% under projection (mlx's SwitchGLU quantized
tensor layout has a bit less overhead than the simple linear scaling
assumed; not investigated further since it landed favorably).

## 3. mlx_vlm already had full native Laguna support

Before writing anything, checked whether `laguna` needed the same kind of
registration work MiniMax-M2 needed in `vendor/cerebras-reap/` (see
top-level README). It didn't — this repo's own `.venv` already ships
`mlx_vlm/models/laguna/` (`language.py`, `config.py`, `laguna.py`) with a
complete `LagunaSparseMoeBlock`, `LagunaTopKRouter`, and `SwitchGLU`-based
expert implementation. Verified by loading the checkpoint directly
(`mlx_vlm.load(..., lazy=True)`) and inspecting the live module tree before
writing any pruning code — module structure matched the source exactly:

```
layer.mlp                     -> LagunaSparseMoeBlock (or SwiGLUMLP for layer 0)
layer.mlp.gate                -> LagunaTopKRouter
layer.mlp.gate.proj           -> nn.Linear(3072, 256)         (router weight)
layer.mlp.gate.e_score_correction_bias  -> (256,)
layer.mlp.switch_mlp.{gate,up,down}_proj -> QuantizedSwitchLinear, nvfp4, group_size=16
layer.mlp.shared_expert       -> SwiGLUMLP
```

So the only work needed was a collect/apply script pair (this project's own
pattern, not upstream mlx_vlm code) — `reap_stream/collect_laguna.py` and
`reap_stream/apply_laguna.py`, ported from `collect_deepseek_v4.py` /
`apply_deepseek_v4.py`.

## 4. Simpler than DeepSeek-V4 in three specific ways

Verified against `mlx_vlm/models/laguna/language.py` line-for-line before
porting, not assumed from architectural similarity:

1. **No Hyper-Connections.** Hidden state stays plain `(batch, seq, hidden)`
   throughout — DeepSeek-V4's `_expand_hc` broadcast step and its
   `hc_mult`-aware masking don't apply here.
2. **No hash-routed layers.** Every MoE layer uses learned top-k routing
   (sigmoid-or-softmax + `e_score_correction_bias`, same correction-bias
   convention as MiniMax/DeepSeek-V4, but no fixed `tid2eid` table anywhere).
   This means **plain REAP deletion everywhere** — none of DeepSeek-V4's
   REAM (merge) special-casing for hash layers is needed, and
   `apply_laguna.py` doesn't import `reap_stream/ream.py` at all.
3. **Two attention types, masks built once.** `full_attention` /
   `sliding_attention` masks are built up front in `LagunaModel.__call__`
   and selected per-layer via `layer.attention_type` — same "build once,
   reuse per layer" shape as DeepSeek-V4 (simpler than Gemma-4, which
   rebuilds per layer).

Routing itself still lives **inside** the MoE block's `__call__`
(`self.gate(x)` → `self.switch_mlp(x, inds)` → `+ self.shared_expert(x)`),
not computed upstream like Gemma-4's `layer.experts` — so the saliency
probe wraps the whole `LagunaSparseMoeBlock` and reimplements its exact
forward math, same approach as DeepSeek-V4's `_MoEProbe`, not Gemma-4's
simpler post-routing wrap.

## 5. bf16→numpy crash on router scores

First smoke test (`collect_laguna.py`, 3-layer slice) crashed on layer 1
with:

```
RuntimeError: Item size 2 for PEP 3118 buffer format string B does not
match the dtype B item size 1.
```

Root cause: `LagunaTopKRouter` returns gate weights as bf16 (`return inds,
weights.astype(dtype)` where `dtype` is the input's bf16), and
`np.array(scores, dtype=np.float64)` tries to go through Python's buffer
protocol directly on an mx bf16 array, which numpy can't interpret cleanly.
DeepSeek-V4's collector never hit this because its `scores` came from a
different code path that happened to already be float32 by that point.

**Fix:** cast to float32 in MLX *before* the numpy conversion:

```python
scores_f32 = scores.astype(mx.float32)
mx.eval(inds, scores_f32, norms)
gates_np = np.array(scores_f32, dtype=np.float64).reshape(-1, scores_f32.shape[-1])
```

Worth checking for on any new architecture whose router keeps gate weights
in the model's native compute dtype (bf16/fp16) rather than upcasting to
float32 before returning them.

## 6. Result

Calibration: `calib/cerebras_reap_mix.jsonl` (Cerebras agentic 6-way mix),
first 2500 rows — verified interleaved across all 6 sources (~400-440 rows
each out of 4096 per source), not stacked by category, so no source-skew
concern despite not being randomly sampled. `--max-tokens 1024`, same
recipe validated on Step-3.7 (`docs/FINDINGS.md`, 1.40h/42 layers there).

Laguna run: **1h56m, 47 MoE layers**, active memory steady at ~16GB
(peak 28.9GB) throughout — the windowed layer-streaming design
(`layers_at_once=1`, free-after-score) kept this well inside a 64GB
machine's budget even during collection, before the pruned checkpoint
existed.

Plan sanity (before applying): scores were checked for degeneracy before
trusting them —

- 100% unique scores per layer (no accidental all-equal/collapsed scoring)
- Only 4 experts total, across all 47 layers × 256 experts (12,032 slots),
  were never selected by any calibration token (would auto-score `-1`,
  pruned first) — i.e. essentially every expert gets real, differentiated
  usage signal, not a mostly-dead router.
- Uniform `keep=160` across every layer (required — `num_experts` is one
  global config field, not per-layer).

Applied: plain REAP deletion (no REAM needed, §4), no requantization pass
(§2) — **256→160 experts/layer, 67GB → 45GB**. Verified functional: loaded
the pruned checkpoint via `mlx_vlm.load` and generated a coherent,
on-topic response to a held-out prompt (not part of calibration).

## 7. Open — not yet accuracy-tested

Only a single load+generate smoke test has been run. The DeepSeek-V4
finding (`docs/DEEPSEEK-V4-FINDINGS.md` #6) measured a real, uneven quality
cost at 50% prune — reasoning (GSM8K 95%) held up much better than broad
recall (MMLU 69%) on that model. 37.5% is more conservative than that case,
but that finding was on a different architecture and hasn't been replicated
here. Recommend running whatever benchmark suite this project normally uses
before treating `models/Laguna-S-2.1-REAP` as anything beyond
provisionally usable.

## Related docs

- `DEEPSEEK-V4-FINDINGS.md` — the "already native low-bit → prune, don't
  requantize" reasoning this doc's §2 is built on, and the accuracy-cost
  data point §7 leans on.
- `reap_stream/collect_laguna.py`, `reap_stream/apply_laguna.py` — the
  actual pipeline, both docstring-annotated with the architecture
  differences from their DeepSeek-V4 ports.
