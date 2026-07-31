# Step-3.7 QLoRA Fine-Tune: Setup & Measurement Plan

Everything staged to fine-tune `Step-3.7-p15-4bit-vblend-shared8` on verified
agent traces. **Nothing has been trained yet** -- this documents the ready
pipeline so the run is a deliberate, measured step, consistent with the rest
of this project.

## Data

Two CC-BY-4.0 datasets of verified coding/tool-use/debugging agent
trajectories, combined:

| source | teacher | trajectories | rows | verifier |
|---|---|---|---|---|
| `greghavens/fable-5-...` | Claude Fable 5 | 2,443 | 13,366 | deterministic tests / tool-arg checks |
| `greghavens/gpt-5.6-sol-...` | GPT-5.6 Sol | 1,474 | 17,939 | tests + model-judge |
| **combined** | 2 teachers | **3,917** | **31,305** | |

"Rows" are cumulative-context slices of a trajectory at each assistant step
(standard agent SFT), so real independent-task count is the **3,917
trajectories**, not 31k rows.

Built by `scripts/build_lora_data.py` into `data/lora_step37/{train,valid,
test}.jsonl` in mlx_lm's native ChatDataset format. Key transforms (each
verified, see the script's docstring):

- Tool-call `arguments` parsed JSON-string -> dict, so Step-3.7's template
  (`arguments | fromjson`) renders under HF's `apply_chat_template` (which
  lacks a `fromjson` filter). Without this, 100% of Fable-5 rows fail to
  render; with it, 0% fail.
- Reasoning level mapped to Step-3.7's three levels
  (`max`/`xhigh`->`high`, `medium`->`medium`, `low`->`low`) and injected as
  `Reasoning: <level>\n\n` at the front of the system message -- the SAME
  channel as the runtime workaround (the Pi `/high` extension), so training
  and inference see identical conditioning. **Level mix is heavily skewed:
  ~31,206 high / 55 medium / 44 low.** This LoRA can therefore reinforce
  strong high-effort behavior but has almost no signal to teach genuine
  low/medium brevity -- treat reasoning-level calibration as out of scope
  for this run (a separate contrastive dataset would be needed).
- **Split is by `source_trajectory_id`, never by row** -- random row
  splitting would leak near-duplicate context across train/valid and produce
  a falsely optimistic val curve. ~85/10/5 train/valid/test by trajectory.
- Rows tokenizing beyond `max_seq_length` are dropped as a memory guard.

### The max_seq_length decision (real tradeoff, worth a look before the run)

Measured token-length distribution across all 31,305 rows:
**p50 = 6,486 · p90 = 48,636 · p99 = 98,226 · max = 214,411.**
These are large cumulative agent trajectories -- later assistant-steps carry
the whole session so far.

At **`max_seq_length = 8192`** the cap drops **42.9%** of rows (13,435),
keeping 17,870 -> **train 15,214 / valid 1,763 / test 893**. That's a real
data-coverage loss, and it's biased: the dropped rows are the deep,
long-context steps -- arguably the most valuable "long agentic loop" examples.

The tradeoff is memory: seq length is the dominant activation-memory lever on
a 93GB resident model. Options:
- **8192 (current):** safe starting point, but loses 43% and undersamples
  long-context reasoning.
- **16384 / 32768:** recovers much of the p50-p90 range, but the smoke test
  must confirm it fits (watch `footprint`, not mx.get_peak_memory).
- The smoke run is exactly where to settle this -- start at 8192, and if peak
  memory has comfortable headroom, rebuild at a higher cap before the full run.

## Config: `artifacts/lora/lora_config.yaml`

QLoRA (LoRA on the already-4-bit model -- the only feasible local path, no
BF16 base exists). The non-obvious choices, each tied to something measured
on THIS checkpoint:

- **LoRA targets the dense, every-token-active pathways only**: attention
  (q/k/v/o/g_proj), dense-layer MLPs (layers 0-2), the MoE **shared expert**,
  and the **router** -- explicitly NOT the 245 routed experts
  (`switch_mlp.*`). Routed experts each see only top-8/288 of tokens, so LoRA
  there is 245x the params for diluted gradient; the shared expert/attention
  see every token. Also aligns with the shared8 tomography finding that the
  always-on non-routed weight class is where this model is most malleable.
- **`num_layers: -1`** (all 45) -- tomography showed sensitivity is spread
  across depth, not localized.
- **`learning_rate: 2e-5`, conservative** -- DWQ on this exact checkpoint
  diverged at 1e-4 and only stabilized at 1e-6/1e-7. This is the #1 thing to
  watch; raise toward 5e-5 only if loss barely moves and stays stable.
- **`batch_size 1`, no grad accumulation** -- DWQ measured grad-accum
  *raising* peak memory (115-130GB) and tripling step time here.
- **`mask_prompt: true`** -- loss only on the final assistant turn.
- **`grad_checkpoint: true`** -- memory guard.
- LoRA rank 16, scale 20 (mlx_lm's own default convention; its `scale` is a
  direct multiplier, not HF's alpha/rank).

## Running (when ready -- NOT yet done)

```bash
scripts/run_lora.sh smoke          # ~30 iters, validate pipeline + memory
scripts/run_lora.sh full 1         # 1 epoch, iters computed from train rows
scripts/run_lora.sh resume         # resume from last checkpoint
```

The launcher runs under `caffeinate` and samples real `footprint <pid>`
memory every 30s (mx.get_peak_memory under-reports >2x here). Adapters +
checkpoints land in `artifacts/lora/adapters/`.

## How to measure (three layers, increasing cost)

1. **Live val loss (built in).** `steps_per_eval: 50` evaluates loss on the
   trajectory-held-out valid split during training. Primary overfitting/
   divergence signal -- watch for the DWQ signature (loss climbing 2-5x above
   its start = stop).
2. **Forgetting check (adapter-aware, no fusing needed).** `mlx_vlm.load`
   takes `adapter_path`, and both eval harnesses now accept `--adapter-path`:
   ```bash
   .venv/bin/python -m reap_stream.eval_ppl_streamed \
     --model <shared8> --adapter-path artifacts/lora/adapters \
     --out artifacts/lora/ppl-lora.json --n-prompts 500
   .venv/bin/python -m reap_stream.eval_multimodal_nll \
     --model <shared8> --adapter-path artifacts/lora/adapters \
     --out artifacts/lora/mmeval-lora.json
   ```
   Compare against the shared8 baselines (**text PPL 5.930**, **multimodal
   NLL 8.390**). A LoRA that improves coding/tool-use while degrading the
   general categories (agentic, reasoning_math, general_instruction) is not a
   win -- this catches that.
3. **Task pass-rate (the real metric, most work).** Both datasets ship with
   verifiers (deterministic tests for coding, tool-call/argument-constraint
   checks for instruction-following). The most direct signal of "did this
   actually help" is running the fine-tuned model's own outputs on held-out
   `test.jsonl` tasks through that same verification -- loss going down does
   not guarantee tool calls stay well-formed or code still runs. This harness
   is not built yet; it's the highest-value follow-up.

## Disk note

~49 GB free at setup. The LoRA itself is cheap (adapters are MB, data <1.1GB).
A later *fused* export would need ~93GB and is not possible without freeing
space first -- but fusing is not required for training or for the
adapter-aware evals above.
