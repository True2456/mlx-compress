# Layer looping & tying notes (Step-3.7)

Research map for [Painters](https://arxiv.org/abs/2407.09298), [Tying the Loop](https://arxiv.org/abs/2606.16825), [Growing→Looping](https://arxiv.org/abs/2602.16490), [T²MLR](https://arxiv.org/abs/2607.15178), [LoopMoE](https://arxiv.org/abs/2606.04438) applied to Step-3.7-Flash + REAP.

## Step layer map (freeze edges)

| Band | Layers | Action |
|------|--------|--------|
| Prelude | 0–2 dense | Never skip / tie / loop |
| MoE | 3–44 (42× routed experts) | REAP; optional tie/loop in **interior** |
| Coda | MTP ×3 + vision | Freeze |

## Phase 1 measurement (148B MLX, already REAP’d 212 experts)

Harness: `python -m reap_stream.cli intervene` → [`artifacts/step37-interventions/interventions.json`](../artifacts/step37-interventions/interventions.json)

Setup: middle band **[12, 36)**, 8 Cerebras-mix prompts × 64 tokens, teacher-forced mean NLL.

| Variant | mean NLL | Δ vs baseline | Note |
|---------|----------|---------------|------|
| baseline | 3.40 | 0 | peak ~79 GB |
| skip every other in middle | 4.00 | **+0.60** | 12 layers dropped |
| reverse middle | 12.06 | **+8.66** | order critical |
| repeat center layer ×3 | 3.41 | **+0.008** | almost free |

### Takeaways vs Painters

- **Order matters hard** on Step (reverse collapses) — aligns with Painters’ math/reasoning sensitivity.
- **Skip hurts** but is survivable at this severity — middle is somewhat canvas-like, not free to delete wholesale.
- **Middle-repeat did *not*** hurt more than skip (unlike Painters’ “repeat one painter” doom). Likely MoE + residuals absorb a single-layer triple; not a green light to replace the whole middle with one block.
- Locked diagnostic band for later work: **middle_start=12, middle_end=36** (keep 3–11 and 36–44 as soft prelude/coda inside MoE).

## Refusal vs benign expert routing (smoke)

Harness: `python -m reap_stream.cli expert-diff`

On 148B MLX, MoE layers **12–35**, 8 refusal + 8 matched-benign prompts × 96 tok:

- Routing **does** shift under refusal (L1 share distance ~0.22–0.30 per layer).
- Largest shifts: layers **30, 35, 21, 29** (not a single “refusal layer”).
- Top single-expert Δshare is small (~1–3% of hits) — **diffuse pattern**, not one expert.
- Correlational only; see `artifacts/step37-refusal-diff/`.

Already runnable:

```bash
# re-run / tweak band
.venv/bin/python -m reap_stream.cli intervene \
  --middle-start 12 --middle-end 36 \
  --max-samples 16 --max-tokens 96 \
  --dataset-file calib/cerebras_reap_mix.jsonl
```

Still waiting on BF16:

1. Full REAP (288→250/216)
2. Post-REAP **expert tying** (`g=2`) using REAP-guided keep (Phase 2)
3. Inference **middle-block loop** k=2 (Phase 3) — reverse result says loop *direction* / block choice matters; start with small interior block (e.g. 20–28), not reverse

## Deferred

- **T²MLR** — needs fusion path + short FT
- **LoopMoE** — IterAdaLN, train-from-scratch
