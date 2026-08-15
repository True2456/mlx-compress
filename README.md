# mlx-compress

Prune, merge, quantize and measure large language models on Apple Silicon
unified memory.

Everything here exists because of one constraint: on a Mac there is no separate
host and device memory pool. You cannot "keep the model in CPU RAM and stream
blocks to the GPU", because there is only one pool. A 155 GB checkpoint on a
128 GB machine cannot be loaded, let alone loaded twice, so every tool in this
repo is built to work a block at a time.

Concretely, that has meant collecting REAP saliency for a 230B MoE, calibrating
an importance matrix for a 155 GB model in 6.3 GB, benchmarking that model on
MMLU without ever holding it in memory, and running distillation across two
Macs over Thunderbolt because one machine could not hold the backward pass.

## What is here

### Collect

| Tool | What it does |
|---|---|
| `collect_imatrix_streamed.py` | oQe-format importance matrix, block by block. 155 GB model calibrated in **6.3 GB** |
| `collect_imatrix_qwen35.py` | Same output for models that do fit; uses the model's own forward |
| `collect_step3p7.py`, `collect_deepseek_v4.py`, `collect_ling3.py`, `collect_laguna.py` | REAP router saliency, per architecture |
| `collect_vision_saliency.py`, `collect_loop_saliency.py` | Saliency for vision towers and latent-loop variants |

### Quantize

| Tool | What it does |
|---|---|
| `awq_quantize_deepseek_v4.py` | AWQ scoped to `switch_mlp`, for Hyper-Connections + hash-routed MoE |
| `awq_quantize_deepseek_v4_distributed.py` | The same across two machines |
| `awq_quantize_qwen35.py` | Sequential AWQ for dense `qwen3_5`, with a bit-exact assertion that the hand-split block forward matches the model's own |
| `awq_quantize_ling3.py`, `quantize_ling3_mixed.py` | KDA/GDN hybrids |
| `quantize_mtp_qwen35.py`, `quantize_mtp_experts.py` | Speculative-decoding drafter heads |
| `dwq_train_student_*.py`, `merge_dwq_distributed.py` | Distillation-aware quantization, single and two-machine |

### Apply

| Tool | What it does |
|---|---|
| `apply_step3p7.py`, `apply_deepseek_v4.py`, `apply_laguna.py`, `apply_gemma4.py` | Write a pruned/merged checkpoint from collected saliency |
| `ream.py`, `tiered.py` | Merge-instead-of-prune, and tiered expert budgets |
| `carry_mtp_weights.py`, `step3p7_mtp_patch.py`, `attach_step3p7_mtp.py` | MTP head surgery |
| `kda_safe_gate_patch.py`, `bailing_swiglu_clamp.py` | Architecture-specific numerical fixes |

### Measure

| Tool | What it does |
|---|---|
| `mmlu_streamed.py` | Likelihood-scored MMLU, block by block, so a 155 GB model can be benchmarked on 128 GB |
| `eval_ppl_streamed.py`, `eval_perplexity.py` | Perplexity, streamed and resident |
| `eval_repetition.py` | Agentic looping / degenerate-repetition probes |
| `eval_mtp_accept_rate.py` | Speculative draft acceptance |
| `divergence_rate_by_length.py`, `localize_divergence.py` | Where a quantized model starts diverging from its parent |

## Models covered

Step-3.7-Flash, DeepSeek-V4-Flash-0731, Qwen3.5/3.8, Ling-3.0-Flash
(`bailing_hybrid`), Laguna-S-2.1, Gemma-4, MiniMax-M2.

## Results worth knowing before you use any of this

Full detail in `docs/`, particularly `DWQ-DISTRIBUTED-FINDINGS.md`.

**Quantizing from bf16 is a different regime from requantizing.** Qwen3.8-27B at
4.85 bpw loses 4 questions out of 564 against its bf16 parent. DeepSeek-V4 at
2.91 bpw, requantized from an already-mxfp4 checkpoint, loses 17 MMLU points.
Same pipeline.

**AWQ calibration is load-bearing, not a refinement.** The same recipe without
it scores 28.5% on MMLU — barely above the 25% chance floor — and emits
gibberish. Both stages matter and they overlap: scale alone removes 57% of the
error, clip alone 61%, together only 65%.

**DWQ recovered nothing.** Held-out KL fell 89–91% and MMLU did not move
(p=0.65, 24 questions gained against 20 lost). It also raised P(EOS) at agentic
turn starts from 0.0004 to 0.0795. Distributional proxies over-credit; judge on
exact-answer or behavioural evals.

**Bit allocation does not transfer between architectures.** On DeepSeek-V4,
`down_proj` is the outlier-concentrated tensor and earns extra bits. On
Qwen3.8 it is the *flattest* tensor in the model, and attention q/k/v is 135×
more concentrated. Copying the recipe across would have spent bits on the
family that needed them least.

**Several plausible improvements measured worse**, and the scripts are kept so
they are not attempted again: masking pad tokens out of AWQ calibration
(`test_awq_padding_mask.py`), feeding the clip search its true scaled-domain
input (`test_awq_stage_ablation.py`), and using an imatrix to improve AWQ
(which already computes the same statistic, and then measures real output error
rather than trusting it).

**Confirm per-layer findings at depth.** An apparent defect at layer 1 reversed
completely by layer 40, where absolute errors are ~1000× larger.

## Layout

```
reap_stream/   collect, quantize, apply, measure (importable, run with -m)
scripts/       launchers, dataset builders, one-off experiments
docs/          findings per model family, with the measurements behind them
vendor/        upstream CerebrasResearch/reap plus architecture patches
artifacts/     model cards and upstream patches (large outputs are gitignored)
calib/         calibration corpora (gitignored)
```

Most tools are run as modules:

```bash
PYTHONPATH=/Applications/oMLX.app/Contents/Resources .venv/bin/python \
  -m reap_stream.collect_imatrix_streamed --model <path> --out imatrix.npz
```

## Related

- [mlx-reap-streaming](https://github.com/True2456/mlx-reap-streaming) — the
  streaming REAP collector, packaged standalone
- [streaming-dwq-mlx](https://github.com/True2456/streaming-dwq-mlx) — the DWQ
  pipeline, packaged standalone
- [jundot/omlx#2657](https://github.com/jundot/omlx/pull/2657) — a prefill
  kernel regression found while benchmarking these builds
