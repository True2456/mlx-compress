# Handoff — state as of 2026-08-16

Where everything is, what is done, what is open. Start here.

> **2026-08-16.** Both published Qwen3.8 checkpoints shipped a **raw-HF MTP head
> on an MLX backbone**, which made MTP a net loss (22.4 tok/s against 24.6 with
> it off). Fixed with `reap_stream/fix_mtp_norms_qwen35.py`, re-uploaded to both
> HF repos, and promoted into the local models — decode is now 43.7–47.3 tok/s.
> A standalone engine, `q38`, was built around it. Full write-up in
> `docs/Q38-ENGINE-AND-SPECULATIVE-FINDINGS.md`; read that before touching MTP,
> DFlash, DSpark or SpecPrefill.

---

## Published models

| model | where | state |
|---|---|---|
| DeepSeek-V4-Flash-0731 AWQ 2/3-bit | [HF](https://huggingface.co/True2456/DeepSeek-V4-Flash-0731-AWQ) | live, card restored 2026-08-15 |
| DeepSeek-V4-Flash-0731 DWQ | — | **repo does not exist**; card written at `artifacts/hf_card_DWQ.md`, unpushed. The AWQ card links to it, so that link is dead |
| Qwen3.8-27B AWQ 4.85bpw | [HF](https://huggingface.co/True2456/Qwen3.8-27B-AWQ-4.85bpw) | live, weights + card uploaded |

**The AWQ model card gets overwritten by weight uploads.** The DeepSeek source
checkpoint ships its own `README.md`, so `hf upload` of a weights folder
clobbers yours. Upload with `--exclude README.md`, or push the card last.

## Code repositories

| repo | contents |
|---|---|
| [True2456/mlx-compress](https://github.com/True2456/mlx-compress) | main working repo (renamed from `step37-reap`). Everything: collect, quantize, apply, measure |
| [True2456/mlx-reap-streaming](https://github.com/True2456/mlx-reap-streaming) | public package: streaming REAP saliency + streaming imatrix (`reap_streaming/imatrix.py`) |
| [True2456/streaming-dwq-mlx](https://github.com/True2456/streaming-dwq-mlx) | private package: DWQ pipeline incl. distributed trainer + streaming MMLU |
| [jundot/omlx#2657](https://github.com/jundot/omlx/pull/2657) | our upstream PR, open |

`streaming-dwq-mlx` and `mlx-compress` contain **near-duplicate copies** of
several collectors. Deduplicating means making one depend on the other; not
resolved.

## Documentation

| doc | covers |
|---|---|
| `docs/Q38-ENGINE-AND-SPECULATIVE-FINDINGS.md` | the `q38` engine, the MTP norm defect and its fix, MTP vs DFlash vs DSpark vs SpecPrefill with measurements, measurement traps |
| `docs/QWEN38-FINDINGS.md` | Qwen3.8-27B: imatrix, recipe, MTP, oMLX kernel bug, compression ladder |
| `docs/DWQ-DISTRIBUTED-FINDINGS.md` | DeepSeek-V4: §1–12 distributed DWQ, §13 approach comparison, §14 bit allocation, §15 streamed imatrix, §16 padding-mask negative result, §17 AWQ stage ablation |
| `docs/DEEPSEEK-V4-FINDINGS.md`, `DEEPSEEK-V4-AWQ-MODEL-CARD.md` | DeepSeek build detail |
| `docs/LING3-*.md`, `LAGUNA-REAP-FINDINGS.md` | Ling-3.0, Laguna |
| `README.md` | repo overview, organised by function, leads with results |

---

## Environment gotchas that cost real time

**The project venv is Python 3.14; oMLX's extensions are cpython-311.** Testing
oMLX native kernels from `.venv` reports "not built" when they are fine. Use
`/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3.11`
with `PYTHONPATH="$O:$O/Python/framework-mlx-base/lib/python3.11/site-packages"`.

**oMLX prefill regression is live** (PR #2657 not merged). Launch with
`~/bin/omlx-fast`, or Qwen prefill drops from 925 to 513 ppTPS at 4k. Verify
with a single `pp 4096 / tg 128`: ~925 means active, ~513 means not.

**`launchctl setenv` does not reach GUI apps** from this shell — different
bootstrap namespace. Launch from Terminal with the env inline instead, and check
`ps -o ppid=` to confirm which way an app was started before trusting an A/B.

**`ps eww` cannot read another process's environment** on this macOS. It returns
nothing even for your own child processes. Do not use it to verify env vars.

**oMLX updates wipe bundle patches.** The `biases` fix is now upstream in
0.6.0.dev1 so nothing needs reapplying, but `artifacts/omlx_patches/` holds the
patches if that changes.

**`/tmp` gets cleared.** Wrapper scripts live in `~/.dwq/` and `~/bin/`.

---

## Method lessons that generalise

These cost days to learn and are the most reusable thing here.

**Proxies over-credit, consistently.** DWQ cut held-out KL 89–91% and moved
MMLU not at all (p=0.65). Distributional and reconstruction metrics repeatedly
predicted improvements that did not appear. Benchmark before believing anything.

**Bit allocation does not transfer between architectures.** DeepSeek's
`down_proj` is the concentrated tensor (PR 0.27); Qwen's is the flattest
(PR 0.2986) with attention q/k/v 135× more concentrated. 94 seconds of imatrix
collection was the difference between a designed and a guessed recipe.

**Confirm per-layer findings at depth.** An apparent defect at layer 1 reversed
completely at layer 40, where absolute errors are ~1000× larger. Weight by
absolute error, not percentage.

**These search heuristics are not estimators.** Four separate attempts to feed
AWQ a *more faithful* input all measured worse: masking pad tokens, using an
imatrix, and both directions of a clip-domain correction. `x_max` only has to
produce a well-conditioned scale vector; the search then measures real output
error. Reasoning about what it "should" see does not predict what works.

**Silent-wrong beats loud-wrong as a failure mode here, repeatedly.** Zeroed MTP
norms, double-shifted layernorms, `vars()` on an MLX Module finding nothing, a
recipe key matching the checkpoint tensor name instead of the module path — all
produced models that loaded and generated fluent text. Assert bit-exactness
where you can (`awq_quantize_qwen35.py --verify-forward`).

---

**Never benchmark with another engine resident.** This cost three separate wrong
conclusions on 2026-08-16. `omlx-server` at 89% CPU made MTP read 12.0 tok/s
instead of 39.4 and produced a confident, entirely false verdict on DFlash.
Resident-but-idle (~1% CPU) is fine; actively working is not. Check
`ps aux | grep omlx-server` before believing any number, and stop `q38` before
benchmarking oMLX.

## Open work, roughly by value

1. **Compression ladder for Qwen** (`QWEN38-FINDINGS.md` §9). Variant A
   (`down_proj` → 2-bit, 15.31 GB) is the principled first cut and a real test
   of whether concentration predicts bit tolerance. Cheapest probe first:
   reconstruction error at 2-bit per family, at depth.
2. **`mtp_num_draft_tokens` > 3** — d3 acceptance was 96%, so the default
   ceiling is likely leaving speed on the table. Config change, no rebuild.
3. **DWQ HF repo** — create `True2456/DeepSeek-V4-Flash-0731-DWQ` and push
   `artifacts/hf_card_DWQ.md`, or strip the dead sibling link from the AWQ card.
4. **gs64 MLP rebuild for Qwen** — measured 45% faster than gs128 in generic
   MLX, and would match the NAX tile. ~85 min, +0.15 BPW. Only worth it if
   #2657 stalls, since the env var already recovers the speed.
5. **Vision + MTP have no imatrix data** — text-only calibration never runs the
   tower. Their bits are guesses. `calib/cloud_reap_8k.jsonl` has 500 rows with
   real images on disk if you want to fix that.
6. **Dedupe `streaming-dwq-mlx` vs `mlx-compress`**.

## Deliberately not doing

- **DWQ on anything else.** It recovered nothing measurable and introduced an
  EOS regression. The failure was method-level (KL over-credits), not
  model-specific.
- **oQ/oQe as a quantizer.** Measured worse *and* bigger than AWQ (44.0% at
  118.4 GB), though that comparison was confounded — no imatrix, borrowed
  sensitivity, MTP stripped. It is a floor, not a verdict.
- **3-bit anywhere in a Qwen MLP.** No kernel; silently drops prefill.
- **Restructuring `reap_stream/` into subpackages.** Every docstring and doc
  uses `python -m reap_stream.X`; the churn outweighs the tidiness.
