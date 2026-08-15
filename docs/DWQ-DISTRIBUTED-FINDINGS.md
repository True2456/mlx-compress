# Distributed DWQ for DeepSeek-V4-Flash: Pipeline, Findings, and Failure Analysis

**Goal:** run DWQ (distillation-aware weight quantization) phase 2 against the
`DeepSeek-V4-Flash-0731-awq2bit3bit-v2` student, distilling from the native
teacher, to recover quantization damage — in particular the agentic
repetition/looping the AWQ v2 recalibration measurably worsened (6/8 → see
`DEEPSEEK-V4-AWQ-MODEL-CARD.md`).

**Hardware:** M5 Max 128GB (rank 0) + M3 64GB (rank 1), Thunderbolt bridge.

**Scope:** every number below is measured on these machines (2026-08-13/14),
not asserted. Three complete training runs were performed; **all three
produced a model that must not be shipped**. The pipeline itself works and is
reusable; the objective was the problem. Both are documented in full because
the failure analysis is the more valuable half.

---

## TL;DR

| # | Finding | Magnitude | Status |
|---|---|---|---|
| 1 | 2-machine DWQ with cross-machine backward works | 100 steps, 0 failures | ✅ built |
| 2 | Quantized-MoE backward costs **~26GB/layer**, stacks linearly | 29 layers ⇒ ~750GB in one graph | ✅ measured |
| 3 | `mx.checkpoint` does **not** help that | identical to the byte | ✅ measured |
| 4 | Chunked backward (separate `mx.eval`'d vjps) fixes it | 119GB peak, fits | ✅ built |
| 5 | Sequence length is nearly **free** here | 768→1024: +0.5GB, +0.2s | ✅ measured |
| 6 | KL fell ~89% in all 3 runs | and told us nothing | ⚠️ trap |
| 7 | Benchmarks flat vs v2 | all within noise | ✅ measured |
| 8 | **Agentic generation regressed** | P(EOS) 0.0004 → 0.33 | ❌ root cause found |
| 9 | Cause = decision-critical **support** omission, not data | arXiv 2607.07050 | ✅ diagnosed |
| 10 | **Quantization costs 17pp of MMLU** — the prize is large | teacher 81.5% vs v2 64.5%, p=3e-6 | ✅ measured |

---

## 1. What was built

| file | purpose |
|---|---|
| `reap_stream/dwq_collect_targets_deepseek_v4.py` | phase 1: stream teacher, cache top-k logits (+`logz`, +forced token support) |
| `reap_stream/dwq_train_student_deepseek_v4_distributed.py` | phase 2: 2-machine training, chunked backward |
| `reap_stream/merge_dwq_distributed.py` | merge both ranks' scale shards into a loadable model |
| `reap_stream/build_omlx_raw_format.py` | convert to oMLX raw format (pre-existing; two bugs found, §6) |
| `reap_stream/quantize_mtp_experts.py` | quantize the MTP drafter (pre-existing) |
| `~/.dwq/rank_wrapper.sh`, `~/.dwq/run_resilient.sh` | launcher + auto-restart (see §5 for why both are needed) |

### The one genuinely new mechanism

`mx.distributed.send`/`recv_like` have **no VJP rule** (verified:
`[Primitive::vjp] Not implemented for Send`), so the model's built-in pipeline
path cannot simply be wrapped in `nn.value_and_grad`. Instead the boundary
hidden state is treated as an ordinary differentiable *argument*:

* rank 1 (early layers) computes `h`, sends it as an untracked tensor, later
  receives `dh` and uses it as the seed cotangent for its own `mx.vjp`;
* rank 0 (late layers + head) receives `h` as a plain input and differentiates
  w.r.t. `(h, its params)` in one call, getting `dh` to send back.

Validated against a same-process `mx.value_and_grad` reference in
`reap_stream/test_dwq_distributed_backward.py` before use.

---

## 2. The memory wall, and why the obvious fix fails

Backward through a quantized MoE layer materialises the dequantized
`[256, out, in]` expert weights **and** their gradients. Measured, one layer:

| layers in one vjp | seq 128 | seq 384 | with `mx.checkpoint` |
|---|---|---|---|
| 1 | 26.18 GB | 26.37 GB | 26.17 GB |
| 2 | 52.00 GB | 52.12 GB | 52.00 GB |
| 4 | 103.54 GB | 103.66 GB | 103.54 GB |

Three things follow, all counter-intuitive:

1. **Cost is ~26GB per layer and stacks linearly.** 29 layers in one graph
   would need ~750GB. The early "successful" 29-layer run at seq=64 only
   survived by thrashing **83GB of swap** — it was never fitting.
2. **It is nearly independent of sequence length.** 3× the tokens cost 0.7%.
3. **`mx.checkpoint` is useless here** — identical to the byte. It elides
   *stored forward activations*; this memory is allocated *inside the
   backward*. Do not reach for it.

### The fix: chunked backward

Forward once keeping only per-chunk boundary `h` (~25MB each), then run the
backward one chunk at a time, each in its own `mx.vjp` + `mx.eval` so MLX frees
between them. Peak becomes `~26GB × chunk_layers` regardless of depth.

Verified bit-exact against a monolithic vjp (`--chunk-layers` ≥ total = the
old behaviour): **0.000e+00** difference at N=1. At N=2 the difference is
1.58e-2 relative, which is bf16 rounding of the intermediate cotangent, not a
bug — chunking makes the backward a genuine bf16 pass where the monolithic
version kept some higher-precision intermediates.

Measured on the real model, 33 layers on rank 0:

| | seq 768 | seq 1024 |
|---|---|---|
| backward (33 chunks) | 23.4 s | 23.6 s |
| rank0 peak | 119.0 GB | 119.5 GB |
| rank1 peak | 57.2 GB | 57.4 GB |

**Sequence length is nearly free**, so the original 768 cap was truncating
52.8% of conversations (and 21.9% of the *agent* category specifically) for no
saving at all. 1024 truncates only 15.5%.

---

## 3. Two silent-corruption traps

Both would have produced a plausible-looking run that learned nothing or
learned wrongly. Both are now guarded in code.

### `mx.checkpoint` + closure-captured params ⇒ **all-zero gradients**

```
reference          val=477.0  gW=[[76,126],[115,193]]
captured-closure   val=477.0  gW=[[0,0],[0,0]]     ← correct loss, zero grads
explicit-args      val=477.0  gW=[[76,126],[115,193]]
```
Params must be passed as **explicit arguments** to a checkpointed function.
`_grad_norm_check()` now aborts if every gradient tensor is zero.

### A "0/8 loop-suspect" score can mean the model cannot speak

`eval_repetition.py` scores distinct-n over generated tokens. A model that
emits EOS immediately generates nothing and trivially scores perfectly.
**Always read `n_gen` before the loop count.**

---

## 4. Results of the three runs

All three: 100 steps, `--scales-only` (2885M trainable scale params, 100% of
them in `switch_mlp` — everything else is 8-bit and excluded by the
`bits < 8` filter), Adam, lr 1e-5.

### Held-out KL — improved hugely every time, and meant nothing

| step | run 1 (top-k KL) | run 2 (+rest bucket) | run 3 (+EOS bound, seq 1024) |
|---|---|---|---|
| 0 | 0.8829 | 0.9471 | 1.1593 |
| 25 | 0.1380 | 0.1378 | 0.1405 |
| 50 | 0.1092 | 0.1176 | 0.1166 |
| 100 | **0.0956** | **0.0963** | **0.1099** |

~89% reduction in every run. One of these models could not generate a single
token. **This is the REAM lesson again** (see `REAM-RESULT.md`: a −0.194 NLL
PPL "win" bought zero accuracy): distributional proxies over-credit.

### Benchmarks (run 1, oMLX harness, n=200/200/164, MTP enabled)

| | v2 baseline | DWQ-100 | Δ | 95% CI |
|---|---|---|---|---|
| MMLU | 61.5% | 61.0% | −1 item | ±6.8pp |
| GSM8K | 93.5% | 94.5% | +2 items | ±3.3pp |
| HumanEval | 87.2% | 84.8% | −4 items | ±5.2pp |

All within noise. No capability gain, no capability loss.

### Agentic generation — the regression

Repetition eval, greedy, 8 real multi-turn trajectories, both models through
the identical path:

| | v2 | DWQ run 3 |
|---|---|---|
| probes generating **nothing** | 0/8 | **5/8** |
| tokens on generating probes | 56–300 | 21, 73, 112 |
| loop-suspect | 2/8 | 0/8 *(vacuous)* |

P(EOS) at an agentic turn-start, same prompt and template:

| | P(EOS) | probes with n_gen=0 |
|---|---|---|
| teacher | never rank-1 in 62 sampled positions (mean 6.1e-4) | — |
| v2 | 0.0004 | 0/8 |
| run 1 (top-k KL) | 0.2048 | 7/8 |
| run 2 (+rest bucket) | 0.3337 | 7/8 |
| run 3 (+EOS bound w=1.0) | 0.1444 | 5/8 |

---

## 5. Root cause

**It is not the training data.** Three independent checks:

* EOS is the teacher's top-1 in **0.188%** of positions — ~1 per conversation,
  i.e. exactly one turn-ending each. Not over-represented.
* At the failing context the teacher rejects EOS: **0/62** positions have it
  top-1.
* The starting model (v2) is clean at 0.0004; the damage appears only *during*
  training and responds to an explicit penalty.

**It is decision-critical support omission** (arXiv
[2607.07050](https://arxiv.org/html/2607.07050)). The loss renormalises over
the teacher's top-k, so the other ~129,000 logits receive **zero gradient** and
the student can inflate them for free. We used `topk=128`; EOS was present at
only **70%** of positions. The paper's parallel case: a teacher's top-32 holds
99.99% of probability mass yet contains the behaviour-switching `<tool_call>`
token in only 0.4% of prompts.

**Why the rest-bucket fix (run 2) failed — a useful negative result.** Adding
the teacher's `logz` and a (k+1)-way KL with an "everything else" bucket
constrains the missing *mass*. But the missing mass is ~6e-4, and in forward KL
the term is `rest_t·log(rest_t/rest_s)` — its gradient is scaled by the
teacher's own rest mass, ~1000× too weak to matter. **This is a support
problem, not a mass problem.** No mass-based penalty can fix it.

### The fix (in flight at time of writing)

Aligned with mlx-lm 0.31.3's own `quant/dwq.py`, which uses **k=1024** (we used
128) — and the same renormalised top-k KL, confirming the structure was never
the bug, only `k`:

* `--topk 1024`
* `--force-token-ids 1,128822` — force EOS and `</think>` into the retained
  support at **every** position, whatever their rank, carrying their true
  teacher logits. Literature-supported: restoring important tokens at every
  position beats restoring only at the first.

The `--special-weight` bound from run 3 becomes self-disabling (it only fires
when the token is absent from the top-k, which can no longer happen) and is
retained as a backstop.

---

## 6. Bugs found in existing tooling

| where | bug | symptom |
|---|---|---|
| `build_omlx_raw_format.py` | shards written without the `model-` prefix | `No safetensors found` |
| `build_omlx_raw_format.py` | `config.json` copied from the **raw** checkpoint, dropping the whole `quantization` block | model silently loads as mxfp4 ⇒ `129 parameters not in model` |
| `dwq_collect_targets_deepseek_v4.py` | prerendered prompts double-wrapped by `apply_chat_template` | corrupted multi-turn calib data (same bug as `awq_quantize_deepseek_v4.py` had) |

Also worth knowing: the oMLX raw build is ~10GB larger than the mlx_vlm source
purely because the conversion **adds the MTP drafter** (10.86GB) from the raw
checkpoint — the AWQ checkpoint contains no MTP tensors at all. Everything else
is a restructuring (645 → 99,330 tensors, same bytes).

---

## 7. Operational notes (things that cost hours)

* **Link-local IPs are not stable.** The Thunderbolt address changed
  `169.254.210.199` → `.190.98` mid-session. `mlx.launch` SSHes into *every*
  host including itself, so this becomes a silent total failure. The runner now
  re-derives both addresses per attempt and pre-accepts host keys.
* **`mlx.launch` does not quote the command path** — a wrapper under
  `/Users/true/Desktop/LLM - Reap/` splits at the space. Keep launcher scripts
  under a space-free path (`~/.dwq/`).
* **`/tmp` gets cleared.** It took out both wrapper scripts mid-session. Nothing
  durable belongs there; checkpoints live in `artifacts/`.
* **A wrapper that redirects its own stdout hides startup failures.** 15 silent
  retries produced zero diagnostics. The runner now warns if the rank log is
  still empty after 120s.
* **Checkpointing doubles peak disk** (writes `.tmp` beside the original before
  renaming). 13.3GB × 2 against 19GB free filled the disk mid-run.
* **The ring link drops.** Observed EPIPE → `Too many send/recv errors` at
  step ~41 with no checkpoint written. Checkpoints now include optimizer state,
  so restarts don't reset Adam's moments.

---

## 8. How to use it

### Phase 1 — collect teacher targets (~80 min, ~15GB)

```bash
./.venv/bin/python -m reap_stream.dwq_collect_targets_deepseek_v4 \
  --teacher ~/Desktop/models/DeepSeek-V4-Flash-0731 \
  --dataset calib/dwq_targets_agentic_weighted.jsonl \
  --out artifacts/dwq-targets-v5-k1024 \
  --n-prompts 2458 --max-tokens 1024 --topk 1024 --layers-at-once 1 \
  --force-token-ids 1,128822 --exclude-categories multimodal
```

Memory is `n_prompts × seq × hc_mult × hidden` (58.7GB at 2458×1024) and is the
binding constraint on sequence length — **not** training. Training seq is
hard-capped by collection seq (`pad_len = min(collected, requested)`), so
raising `--train-max-tokens` above the collected value is a no-op.

### Phase 2 — distributed training (~70 min for 100 steps)

Both machines need the repo, the venv, the student model, and the targets.

```bash
scp -q reap_stream/dwq_train_student_deepseek_v4_distributed.py PEER:"/Users/true/Desktop/LLM - Reap/reap_stream/"
scp -q artifacts/dwq-targets-v5-k1024/* PEER:"/Users/true/Desktop/LLM - Reap/artifacts/dwq-targets-v5-k1024/"
```

```bash
~/.dwq/run_resilient.sh artifacts/dwq-v5-ckpt 100 artifacts/dwq-targets-v5-k1024 \
  --layer-split 10,33 --chunk-layers 1 --lr 1e-5 --val-frac 0.05 --eval-n 20 \
  --scales-only --ckpt-every 25
```

* `--layer-split A,B` in **forward** order: position 0 (= rank 1, the smaller
  machine) gets the first A layers; rank 0 gets the last B **plus** the head.
  `10,33` is tuned to the 55.7GB / 123.5GB GPU working-set ceilings — note
  those are well below physical RAM (the M3 caps at 55.7GB, not 64).
* `--chunk-layers 1` — raise only with spare headroom; each layer in flight is
  ~26GB.
* Resumes automatically from `--ckpt-dir`; the runner survives link drops.

### Phase 3 — merge, convert, deploy

```bash
./.venv/bin/python -m reap_stream.merge_dwq_distributed \
  --source models/DeepSeek-V4-Flash-0731-awq2bit3bit-v2 \
  --ckpt-dir artifacts/dwq-v5-ckpt \
  --out models/DeepSeek-V4-Flash-0731-awq2bit3bit-v2-dwqv5
```

Fetch rank 1's shard from the peer into `<ckpt-dir>/rank1/` first. The merge
APFS-clones the source and patches tensor bytes **in place** (identical shape
and dtype), so it costs ~5GB rather than re-writing 101GB. It refuses to
overwrite an existing `--out` without `--force`, and never touches `--source`.

```bash
./.venv/bin/python -m reap_stream.build_omlx_raw_format \
  --awq-checkpoint models/DeepSeek-V4-Flash-0731-awq2bit3bit-v2-dwqv5 \
  --raw-checkpoint ~/Desktop/models/DeepSeek-V4-Flash-0731 \
  --out ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw-v5
```

**Then apply both workarounds from §6** — rename shards to the `model-` prefix
(updating `model.safetensors.index.json` in lockstep) and copy the
`quantization` / `quantization_config` blocks from an existing working oMLX
build's `config.json`. Without these it will not load.

```bash
./.venv/bin/python -m reap_stream.quantize_mtp_experts \
  --src ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw-v5 \
  --out ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw-v5-mtpq
```

Hardlinks unchanged shards, so this costs ~7GB, not another 104GB. Saves
3.82GB on the drafter. Safe to be aggressive: DSpark uses exact rejection
sampling, so drafter error costs accept rate, never correctness.

### Phase 4 — evaluate, **in this order**

```bash
./.venv/bin/python reap_stream/eval_repetition.py \
  --model ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw-v5-mtpq \
  --probes calib/ds4_agentic_repetition_probes.jsonl --max-tokens 300
```

**Read `n_gen` first.** If any probe is 0, the model cannot generate and every
other number is meaningless. Only if it generates:

```bash
PYTHONPATH=/Applications/oMLX.app/Contents/Resources \
./.venv/bin/python -m reap_stream.run_omlx_benchmarks \
  --model ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw-v5-mtpq \
  --benchmarks mmlu gsm8k humaneval --n 200 --batch-size 4 \
  --sampling deterministic --out artifacts/bench_v5.json
```

Baseline to beat: **61.5 / 93.5 / 87.2**. This uses oMLX's own `omlx.eval`
benchmark classes and loader, so results are comparable to the app's.

### A 2-minute sanity check that would have caught everything

```bash
env PYTHONPATH=/Applications/oMLX.app/Contents/Resources \
./.venv/bin/python scripts/dwq_probe_eos.py <model-path> LABEL eb5e23be1ceb
```

Prints the top-5 next tokens at an agentic turn-start. If
`<｜end▁of▁sentence｜>` is rank 1, the model is broken — regardless of what KL,
benchmarks, or the loop score say. Run this before converting anything.

---

## 9. Recommendations

1. **Do not judge a distillation run by KL.** It fell ~89% in all three runs,
   including one that produced no output at all. Gate on an exact-answer or
   behavioural eval on the target distribution.
2. **Check `n_gen` before any generation-quality metric.**
3. **Use `topk` ≥ 1024 and force decision-critical tokens into the support.**
   Small `k` silently removes rare-but-critical tokens from supervision.
4. **Verify gradient *magnitude*, not just formulation.** The rest-bucket was
   correctly derived and practically useless because its gradient was scaled by
   a quantity ~1000× too small.
5. **Sequence length is nearly free here** — do not truncate to save memory;
   collection, not training, is the constraint.


---

## 10. How much is there to recover? (measured 2026-08-14)

The teacher had **never been benchmarked**, so for five training runs the target
was unknown. Measured with `reap_stream/mmlu_streamed.py` (likelihood-scored
A/B/C/D, oMLX's own 5-shot prompt and stratified sample, model streamed
block-by-block so 155GB fits on a 128GB machine):

| | MMLU (n=200, likelihood) |
|---|---|
| teacher `DeepSeek-V4-Flash-0731` | **81.5%** (163/200) |
| student `awq2bit3bit-v2` | **64.5%** (129/200) |
| **gap** | **+17.0pp**, 95% CI [10.1, 23.9] |

Paired on identical questions:

| | count |
|---|---|
| both correct | 121 |
| both wrong | 29 |
| **teacher only correct** | **42** ← recoverable |
| student only correct | 8 |

McNemar chi2=21.8, **p=3.06e-06**. Not noise.

Sanity check on the method: the student scores 64.5% here vs **61.5%**
generation-scored by the oMLX harness — within noise of each other, so
likelihood scoring measures substantially the same thing. Absolute numbers are
still not interchangeable with the generation-scored table in §4; the *gap* is
the meaningful quantity.

**This reframes §4.** Run 1's flat benchmarks do not mean "there was nothing to
gain" — they mean DWQ recovered approximately none of a 17-point deficit,
because support omission had broken the objective. The prize is large and
remains unclaimed.

**Do this measurement first on any future quantization work.** Five runs were
spent optimizing toward a target whose distance had never been established.


---

## 11. Final verdict: DWQ does not recover the deficit (measured 2026-08-14)

MMLU n=200, likelihood-scored, all models on identical questions, paired:

| model | MMLU | vs v2 (paired) |
|---|---|---|
| teacher | **81.5%** | +42/-8, p=3e-06 |
| student v2 | 64.5% | (baseline) |
| dwqv5 (k=1024, EOS bound w=1.0) | 66.5% | +24/-20, **p=0.65** |
| dwqv6 (k=1024, EOS bound w=5.0) | 66.5% | +27/-23, **p=0.67** |

Near-symmetric gained/lost splits: the models churn answers rather than
improve. Teacher still ~30 questions ahead of both. **DWQ recovered ~10% of a
17-point deficit, and that 10% is not distinguishable from zero.**

`dwqv5 vs dwqv6: +5/-5, p=0.75` — the two EOS-bound weights are statistically
identical, as the gradient measurement predicted: the bound only fires when EOS
is ABSENT from the teacher's top-k (`absent = 1 - any(idx==tid)`), and at the
failing turn-start positions EOS coverage is 100%, so `d(loss)/d(EOS logit)` is
**byte-identical (7.949989e-02) at w=0, w=1 and w=5**. Three runs differed only
where the problem wasn't.

### The KL trap, quantified one last time

| | KL (the objective) | MMLU (what matters) |
|---|---|---|
| v2 -> dwqv5 | 1.1652 -> 0.1101, **-91%** | 64.5% -> 66.5%, **p=0.65** |

A 91% improvement on the training objective bought nothing measurable. Third
occurrence of this pattern in the project (see `REAM-RESULT.md`, §4 above).

### What would have to change to make DWQ work here

Not the loss — three variants are provably identical at the failing positions.
The plausible remaining levers, none validated:
* **more steps** — 100 steps over 2335 prompts is ~4% of one epoch;
* **train more than scales** — only 129 expert-scale tensors are trainable
  (everything else is 8-bit and excluded by the `bits < 8` filter), which may
  simply lack the capacity to close a 17pp gap;
* **a different target** — the deficit may live in the 2-bit *weights*, not in
  their scale factors, in which case no scale-only method can reach it.


---

## 12. Is the agentic reasoning better? No. (measured 2026-08-14)

MMLU cannot see agentic behaviour, so this measures NLL of the REAL assistant
continuation on 60 held-out agentic trajectories (5,951 assistant tokens,
`/tmp/agentic_nll.py`, streamed so the teacher fits):

| model | assistant-token NLL |
|---|---|
| v2 | 0.5033 |
| dwqv5 | **0.4921** |
| dwqv6 | 0.4937 |
| **teacher** | **0.5678** ← worst |

**The teacher scores worst, which invalidates this metric as a quality
measure.** The trajectories are real traces from the calibration corpus; v2 was
AWQ-*calibrated* on that distribution and DWQ trained on it further, so the
students are fit to it and the teacher is not. The metric measures corpus fit,
not reasoning — DWQ's 2% gain is the training objective leaking into the
evaluation.

The only *behavioural* agentic measurement is the one DWQ loses on: 1/8 probes
generate nothing and outputs collapse to 22-196 tokens vs v2's 56-300. Better
token-level fit alongside collapsed generation = better at predicting tokens it
will never produce.

**Design implication:** we distilled toward a teacher that models this agentic
corpus *worse than the student already did*. Coherent for recovering
quantization damage (the goal is matching the teacher's function), but it means
this corpus could never teach agentic behaviour via DWQ, whatever the loss.


---

## 13. Quantization-approach comparison (measured 2026-08-14)

All MMLU likelihood-scored, n=200, identical stratified sample, same scorer,
paired question-by-question via `reap_stream/mmlu_streamed.py`.

| build | MMLU | size | vs AWQ (paired) |
|---|---|---|---|
| teacher (native mxfp4/mxfp8) | **81.5%** | 155.0 GB | +42/-8, p=3.06e-06 |
| **AWQ 2/3-bit (shipped)** | **64.5%** | 108.2 GB | — |
| + DWQ distillation | 66.5% | 108.2 GB | +24/-20, **p=0.65** |
| oMLX oQ2.5 | 44.0% | 118.4 GB | -41 questions, **p=1.47e-06** |
| same recipe, **no calibration** | 28.5% | 92 GB | gibberish output |

**The shipped AWQ recipe beat every alternative tried.**

### 13.1 The teacher is itself quantized

The raw HF release ships routed experts at native **mxfp4 (4-bit)** and
attention/shared at **mxfp8 (8-bit)** — verified against the live-loaded model.
So the 81.5% ceiling is a 4-bit model, every build here requantizes *from*
4-bit, and no approach has a "quantize from bf16" advantage. The 17pp gap is
4-bit -> 2/3-bit.

### 13.2 AWQ calibration is load-bearing, not a refinement

A plain-RTN build at the *same* bits and group sizes
(`build_deepseek_v4_quant98.py`) scored **28.5%** — barely above MMLU's 25%
chance floor — and generated literal gibberish:

    '}<?iger}<?}<?iger<?>\n\ncodeline}<?ulumcodeline...'

Reconstruction error was verified at **44.5% relative RMS with matching
magnitudes** (teacher rms 0.02442 vs RTN 0.02454), i.e. textbook 2-bit RTN, not
a broken build. **At 2-bit, without activation-aware scaling, this model
collapses.** Calibration is the difference between a working model and noise —
this justifies the whole AWQ pipeline in a way the v1->v2 comparison never did
(that changed corpus *and* down_proj group size together).

### 13.3 oQ (oMLX's built-in quantizer) — what it is and why it under-performed here

`omlx/oq.py` is a serious mixed-precision quantizer: GGUF K-quant layer
position strategy + unsloth Dynamic 2.0 selective non-quantization + BnB
MSE-optimal clipping, levels oQ2..oQ8, with **DeepSeek-V4-specific handling**
(`_is_deepseek_v4_config`, and an explicit guard that fp16 oQ "can collapse to
repeated BOS tokens" on this model). It also ships `oqe_calibration_data.json`
(4.5 MB; tool_calling/chat/reasoning/code/multilingual).

Three blockers hit on a 128 GB machine, all real:

1. **oQE imatrix cannot run.** Its RAM-safe proxy is **151.5 GB** against a
   90 GB calibration limit — because a uniform 4-bit proxy of an *already*-mxfp4
   model does not shrink. Unfixable by any memory setting.
2. **The 90 GB limit is a proportional 25% reserve**
   (`_MAX_MODEL_RAM_FRACTION = 0.75`) sized for 16/32 GB machines. Patching it
   to 0.88 in-process gave limit=113.4 GB, reserve=15.5 GB, which is what
   allowed the run at all.
3. **VLM vs LLM key shape.** `sensitivity_model_path` must be LLM-shaped; our
   repo-format AWQ carries `language_model.` prefixes and is rejected. The
   oMLX-raw build works.

The resulting build also **stripped the MTP drafter** (114 tensors vs 7009)
despite `preserve_mtp=True`. So 44.0% is a *floor* for oQ, not a fair verdict —
it ran without imatrix, with borrowed sensitivity, and lost speculative
decoding. Not published as a comparison for that reason.

### 13.4 The oQ imatrix format is simple and quantizer-agnostic

Per quantized tensor, two numpy arrays, `np.savez_compressed` + a JSON
metadata sidecar; cache validation is plain metadata key equality
(`_oqe_cache_matches`):

    entry.in_sum2 += np.square(x_np).sum(axis=0)      # dense: [in_features]
    entry.counts[0] += x_np.shape[0]
    entry.in_sum2[expert] += x_sq[rows].sum(axis=0)   # MoE: [n_experts, in_features]
    entry.counts += np.bincount(idx_flat, minlength=n_experts)

Every operation is a running sum over tokens — **no global state, nothing that
needs the full model resident.** A layer-by-layer streaming pass computes
identical statistics at oQ's defaults (128 samples x 512 tokens) in
**~6 GB**: 128*512*hc_mult(4)*4096*2B ~= 2.1 GB of hidden states plus one layer
(~3.6 GB). Two orders of magnitude under the 151.5 GB proxy.

An imatrix is activation statistics, not a quantization method, so the same
file can drive GGUF/llama.cpp, oQ, a custom mixed-precision pass, **or answer
the open bit-allocation question directly** (see §14).

---

## 14. The open question: where should the bits go?

Still unmeasured, and the highest-value remaining experiment.

| | ours | mlx-community 2.4bit-mixed |
|---|---|---|
| gate_proj / up_proj | 2-bit gs128 (26.0 GB each) | ⎫ uniform 2-bit gs128 (77.9 GB) |
| **down_proj** | **3-bit gs64 (40.4 GB)** | ⎭ |
| attention | 8-bit gs64 (5.6 GB) | 6-bit gs128 (4.0 GB) |
| MTP drafter | 2/3-bit (7.04 GB) | 3-bit gs128 (8.4 GB) |
| total | 108.4 GB | 92.6 GB |

**`down_proj` alone is 14.5 GB of the ~16 GB difference.** Our recipe gives it
both more bits and finer groups; theirs treats all three projections alike.
That allocation was inherited from the v2 recalibration and has never been
isolated. A per-channel imatrix over our own agentic corpus answers it directly
— one ~6 GB pass instead of building and benchmarking N candidate models at
~100 GB and ~15 min each.

Also unmeasured and cheap: **gate/up `gs128 -> gs64`** costs only ~1-2 GB
(group size affects scale count, not weights) and targets weight-value fidelity
— which is exactly where §11 showed the damage lives, and where DWQ provably
could not reach.

---

## 15. Streamed imatrix: answering the bit-allocation question

§14 left open where the bits should go. `reap_stream/collect_imatrix_streamed.py`
answers it with measurement.

### 15.1 The tool

oQ's own oQe calibration cannot run on this checkpoint: its RAM-safe proxy is
151.5GB against a 90GB limit, because a uniform 4-bit proxy of an *already*
mxfp4 checkpoint doesn't shrink (§13.3). No memory setting fixes it — machine
capacity is 128.8GB.

But an imatrix is only a running sum of squared input activations per tensor.
No global state; nothing needs the whole model resident. So we stream
block-by-block, reusing oQ's `OQImatrixCollector` / `_save_oqe_imatrix` /
`_source_imatrix_signature` directly so the output is format-identical.

**Measured: 6.3GB peak for a 155GB model, 264s, 596 entries** (467 dense,
129 MoE = 43 layers x 3 projections, full coverage). Verified by round-trip
through oQ's own loader: all finite, non-negative, correct shapes
(`[in]` dense, `[n_experts, in]` MoE).

Trap: `collector.install()` stashes every wrapped module in
`_original_modules`. Freeing `text.layers[li]` releases nothing while those
references live — measured leak 3.4GB -> 98.3GB over 43 blocks. Purge the
layer's entries each step.

### 15.2 Raw energy does NOT compare across projections

| projection | mean energy/channel/token |
|---|---|
| shared_experts (8-bit) | 0.2950 |
| attention (8-bit) | 0.0299 |
| down_proj (3-bit gs64) | 0.0023 |
| gate_proj (2-bit gs128) | 0.0008 |
| up_proj (2-bit gs128) | 0.0008 |

`gate_proj` and `up_proj` read the *same* input, which is why they are
identical to 4 decimals — a correctness check on the collector, not a finding.
`down_proj` reads the post-SwiGLU intermediate: a different vector space.
Cross-space magnitude comparison says nothing about bit allocation.

### 15.3 Outlier concentration is the metric that answers it

Scale-free, comparable across projections, and it is what group-wise
quantization actually struggles with. Over 10,962 routed expert rows:

| projection | top-1% of channels carry | participation ratio |
|---|---|---|
| **down_proj** 3b/gs64 | **16.1%** | **0.2729** |
| gate_proj 2b/gs128 | 3.1% | 0.8368 |
| up_proj 2b/gs128 | 3.1% | 0.8368 |

**`down_proj` is 5x more outlier-concentrated**, with only ~27% of channels
effectively carrying signal versus ~84% for gate/up. Quantization error is
dominated by high-energy channels; when 16% of energy sits in 1% of channels a
coarse group scale is dragged toward outliers and the rest lose resolution.
That is exactly what more bits + smaller groups fix.

**Conclusion: the 14.5GB `down_proj` premium is spent on the one tensor family
the data says needs it** — measured on our own agentic corpus, not inherited.
`gate/up` look genuinely comfortable at 2-bit, consistent with GSM8K 93.5% /
HumanEval 87.2%.

### 15.4 Limits

- This is a **proxy**. Concentration predicts quantization difficulty; it does
  not measure accuracy. The decisive test remains a uniform-2-bit `down_proj`
  build (~93GB) benchmarked paired against the current one.
- 46 of 11,008 expert rows never routed on this corpus and carry no statistics.
- Entry names are checkpoint-shape dependent (`language_model.` prefix on
  VLM-shaped builds). An imatrix collected on one shape needs name remapping
  before oQ will match it against the other.

### 15.5 Reuse

An imatrix is activation statistics, not a quantization method — the same file
drives oQe (`imatrix_cache_path` + `enhanced=True`), GGUF/llama.cpp, or a
custom mixed-precision pass. And the streaming approach generalizes to any
model too large to calibrate in RAM, not just this one.

```bash
PYTHONPATH=/Applications/oMLX.app/Contents/Resources .venv/bin/python \
  -m reap_stream.collect_imatrix_streamed \
  --model ~/Desktop/models/DeepSeek-V4-Flash-0731 \
  --dataset calib/ds4_agentic.jsonl \
  --out artifacts/imatrix_teacher_agentic.npz \
  --num-samples 128 --seq-length 512
```

---

## 16. Padding-mask in AWQ calibration: NEGATIVE result

### 16.1 The hypothesis

`awq_quantize_deepseek_v4.py` right-pads every calibration prompt to
`--max-tokens` so captured activations concatenate. Pad positions are NOT
excluded from the causal mask, so they carry real activations from a degenerate
repeated-pad context.

**Scale caveat on everything in S16/S17:** these were run at `128 x 384`
(19.5% padding). The published v2 build was calibrated at **`--n-prompts 256
--max-tokens 768`** = 30.7% padding, 42.6% of prompts truncated, 136,332 real
tokens -- i.e. 4x the data and a materially different padding fraction. The
signs are consistent and large enough that they are unlikely to flip, but these
results are strictly OFF-production and a re-run at 256x768 would be needed to
claim otherwise.

`search_best_scale` reduces with `x.abs().mean(axis=(0,1))` and scores
`mse(out, out_q)` over every captured position, so the reasoning was: padding
both biases the per-channel magnitudes and dilutes the objective.

### 16.2 Result: masking makes it WORSE

Real-token reconstruction error, `mse(ffn_fp(x_real), ffn_quant(x_real))`,
scoring both scale sets on the SAME real-token objective:

| layer | padded (production) | masked | masking penalty |
|---|---|---|---|
| 1 | 2.451e-03 @ ratio 0.70-0.75 | 2.879e-03 @ ratio 1.50 | **+17.5%** |
| 5 | 2.807e-03 @ ratio 0.70 | 3.005e-03 @ ratio 1.10 | **+7.0%** |
| 20 | 3.987e-02 @ ratio 0.50 | 4.162e-02 @ ratio 0.60 | **+4.4%** |

Consistent sign at every depth. `--mask-padding` therefore ships **off**;
the flag and plumbing are kept so the negative result can be re-run.

### 16.3 Why the reasoning was wrong

`x_max` is **not an estimate of anything** -- it is a conditioning heuristic.
Nothing requires it to be a faithful statistic of real tokens; it only has to
produce a scale vector that quantizes well, and the grid search then measures
*actual output error* to pick the exponent. "More faithful input statistic"
and "better scales" are different things.

This is the same lesson as S13/S15 from a new angle: a proxy that looks more
principled is not thereby better. Measure the objective you care about.

### 16.4 A confound that nearly produced a wrong number

The first A/B reported masking as **-54%** on layer 1. That was an artifact:
AWQ hardcodes its exponent grid to `ratio = g/n_grid`, i.e. **[0, 1)**, and the
masked statistic's optimum sits at **1.50** -- outside the searchable range, so
its curve was still descending when the grid ran out. Comparing
"best within [0,1)" across two statistics with different dynamic range
penalises whichever one needs a larger exponent.

Sweeping to ratio 2.4 gave both a fair range and cut the gap from 54% to 17.5%
-- but did not change the sign. `n_grid` controls resolution, NOT range.

**Corollary, checked and negative:** the production (padded) statistic's optima
are 0.70 / 0.70 / 0.50 on layers 1 / 5 / 20 -- comfortably inside [0,1) with
room either side. The shipped v2 build is **not** being clipped by AWQ's grid.

### 16.5 A real trap in the implementation

The first version index-mapped pad positions out of `down_proj`'s captured
activations assuming shape `(P, L, k, 1, inter)` -- which a small probe
confirmed. At production sizes SwitchGLU **sorts by expert and flattens** to
`(P*L*k, 1, inter)`: the layout is data dependent. A fixed index map would have
scrambled token<->expert correspondence on exactly the runs that matter, while
still producing plausible-looking numbers.

The shipped version never maps indices: it replays `layer.ffn` on the
already-masked tensors and lets SwitchGLU emit whatever layout it wants. Worth
remembering for any future work that touches captured MoE activations.

```bash
# reproduce (both the A/B and the wide-grid diagnostic)
PYTHONPATH=/Applications/oMLX.app/Contents/Resources .venv/bin/python \
  -m reap_stream.test_awq_padding_mask --layers 0 1 2 5 10 20 --n-grid 20
PYTHONPATH=/Applications/oMLX.app/Contents/Resources .venv/bin/python \
  -m reap_stream.test_awq_padmask_diag --layer 1 --n-grid 24 --max-ratio 2.4
```

---

## 17. AWQ stage ablation: production recipe is correct (no change)

The 28.5% no-calibration result removed BOTH AWQ stages at once, so it never
established what each contributes. `reap_stream/test_awq_stage_ablation.py`
separates them, scored on real-token FFN reconstruction error:

    rtn      quantize(W)                  no calibration
    scale    quantize(W*s)/s              search_best_scale only
    clip     quantize(clip(W))            search_best_clip only
    both     quantize(clip(W*s))/s        PRODUCTION
    both_fix clip search fed x/s          "domain-corrected" variant

### 17.1 Both stages are load-bearing, and they overlap

Layer 1: gate/up scale -57.5%, clip -60.8%, both -64.9%. Two stages that each
remove ~58-61% of the error combine to only 64.9% -- they are largely fixing
the SAME errors, which caps what tuning either knob can buy. Neither can be
dropped to save calibration time.

### 17.2 Scale spread tracks input concentration (predicted in advance)

gate/up scale spread 1.88x, down_proj 6.66x -- 3.5x apart, in the same block,
from the same code. This is independent confirmation of S15: gate/up's input is
an RMSNorm output and is nearly flat (per-channel mean|x| max/min = 1.7,
p99/p50 = 1.22), while down_proj's is post-SwiGLU and concentrated.

Note this refuted a plausible prediction: a flat input does NOT make the scale
search inert. A 1.88x spread still removes 57.5% of gate/up's error at 2-bit.

### 17.3 The depth reversal -- why no change was made

Layer 1 showed down_proj scale-only (-59.4%) beating production's scale+clip
(-47.2%), suggesting the clip search should be skipped for down_proj. **That
did not generalise.** down_proj, scale vs both:

| layer | scale | both (production) | winner |
|---|---|---|---|
| 1 | 1.980e-03 | 2.577e-03 | scale, by 23% |
| 10 | 1.634e-03 | 1.736e-03 | scale, by 6% |
| 25 | 2.991e-03 | 2.643e-03 | **both, by 12%** |
| 40 | 1.495e+00 | 2.241e-01 | **both, by 85%** |

gate/up mirrors it: both wins at L1/L10/L25, loses to scale-only at L40.

Decisive factor is **absolute** scale: layer 40's errors are ~1000x layer 1's
(gate/up rtn 3.2e+01 vs 5.8e-03). Total damage is dominated by deep layers, and
that is exactly where production's scale+clip is overwhelmingly right. The
"improvement" would have traded 23% on a 1.98e-03 error for an 85% regression
on a 1.50e+00 one.

**Conclusion: production (scale + clip, both families, all layers) is correct.
No change.**

### 17.4 Fidelity-improving changes keep losing

`both_fix` fed the clip search the x/s its scaled weights actually see at
inference, correcting a real domain mismatch (apply_scale runs between capture
and clip). It was **worse on both families** (gate/up 61.8% vs 64.9%; down
29.3% vs 47.2%).

That is the fourth such result: masking padding (S16), the imatrix as a more
principled statistic (S15), and both directions of the clip-domain correction.
These searches are **conditioning heuristics, not estimators** -- reasoning
about what they "should" see does not predict what works. Measure the target
objective; do not reason from first principles about the inputs.

```bash
PYTHONPATH=/Applications/oMLX.app/Contents/Resources .venv/bin/python \
  -m reap_stream.test_awq_stage_ablation --layer 40 --n-grid 20
```
