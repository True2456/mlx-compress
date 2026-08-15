---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - mlx
  - quantized
  - awq
  - deepseek
  - moe
pipeline_tag: text-generation
library_name: mlx
---

# DeepSeek-V4-Flash-0731-AWQ (2/3-bit, MLX/oMLX)

A 2-bit/3-bit AWQ quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) for Apple Silicon, built for [oMLX](https://github.com/jundot/omlx). Targets ~100-112GB depending on whether the embedded MTP/DSpark speculative-decoding head is loaded, down from the ~155GB source checkpoint.

This is **not** a general MLX-compatible export — it requires oMLX's custom DeepSeek-V4 loader (raw per-expert tensor layout, custom `chat_template_v4.py` encoder). It will not load via a plain `mlx_lm.load()` or in LM Studio's stock model picker.

## Quantization recipe

- **Routed experts** (`gate_proj`/`up_proj`): 2-bit affine, group_size=128
- **Routed experts** (`down_proj`): 3-bit affine, group_size=64 (down_proj is more error-sensitive post-SwiGLU-activation; tightened independently of gate/up)
- **Attention / shared experts / router / embed / lm_head / hc_head**: left at native precision (mxfp4/mxfp8, untouched by AWQ)
- **MTP/DSpark drafter** (3 embedded speculative-decoding stages): left at native precision, not yet AWQ'd (~10GB of the total; see Limitations)
- **Calibration data**: [antirez/ds4](https://github.com/antirez/ds4)'s imatrix corpus (`gguf-tools/imatrix/dataset/prompts.jsonl`), 256 shuffled prompts spanning source-code review, real multi-turn agent/tool-call trajectories, language, translation, and reasoning benchmarks — not a single-turn-only set. Full corpus and provenance notes in `reap_stream/convert_ds4_imatrix_prompts.py`.
- Built with `reap_stream/awq_quantize_deepseek_v4.py` (`mlx_lm.quant.awq`'s `search_best_scale`/`search_best_clip`, adapted for DeepSeek-V4's Hyper-Connections + hash-routed MoE architecture) and converted to oMLX's raw format via `reap_stream/build_omlx_raw_format.py`.

## Benchmark results

Deterministic (greedy) sampling, 200 questions each (HumanEval capped at its 164-item full set), batch_size=4, thinking disabled, measured against a prior single-turn-calibrated AWQ build (v1) for comparison:

| Benchmark | v1 (single-turn calib) | v2 (this build) |
|---|---|---|
| MMLU | 34.5% (69/200) | **61.5%** (123/200) |
| GSM8K | 77.0% (154/200) | **93.5%** (187/200) |
| HumanEval | 72.6% (119/164) | **87.2%** (143/164) |

## Reasoning / thinking mode

DeepSeek-V4-Flash-0731 supports three real `reasoning_effort` levels: `low`, `high`, `max` (not `medium`/`xhigh` — those aren't valid and the chat template will reject them). Recommended sampling for agentic use per the base model's own documentation: `temperature=1.0, top_p=0.95`. Pass `enable_thinking`/`reasoning_effort` via `chat_template_kwargs` in the request body.

## Limitations

- **Agentic repetition**: on an 8-probe long-horizon multi-turn repetition eval (real tool-call trajectories, 300-token continuations, checked for degenerate n-gram repetition), this build flags 6/8 as loop-suspect vs. 4/8 for the earlier single-turn-calibrated build — the broader calibration data measurably improved general capability but did not fix, and by this narrow metric slightly worsened, long-horizon coherence under repetitive multi-turn context. This looks like a routing-stability effect that a local/per-layer quantization objective (AWQ) isn't well suited to correct; a full end-to-end distillation pass (DWQ) against the real teacher's output distribution is the planned next step for this specific issue.
- **MTP/DSpark not yet quantized**: the embedded 3-stage speculative decoder is copied byte-for-byte from the source checkpoint at native precision. Estimated ~4-4.5GB of additional savings available by extending the same AWQ treatment to it, not yet done (DSpark's block-prediction forward pass differs structurally from the standard per-layer decode the current calibration script is built around).
- oMLX-specific loader required; not portable to other MLX/GGUF runtimes as-is.

## Files

Standard safetensors shards + `model.safetensors.index.json`, oMLX's raw per-expert tensor layout (not fused `switch_mlp` — matches the source checkpoint's HF-style key naming so oMLX's custom sanitize/quantize pipeline can bind biases correctly). `config.json`'s `quantization` field carries per-tensor-path bit/group_size overrides, merged from oMLX's native-scheme generator (`make_quantization_config`) plus the 129 AWQ-tuned backbone paths.
