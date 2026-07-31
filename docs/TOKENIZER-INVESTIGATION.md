# Tokenizer: numeric corruption — real bug, found and fixed

**Date resolved: 2026-07-26.** Numeric corruption in serving (`kill 18452` →
`kill -9 1845`, `2456, 1337, 495` → `2 4 5 6, 1 3 3 7, 4 9 5`, etc.) was a
real, fixable bug: `tokenizer_config.json` declares
`"tokenizer_class": "LlamaTokenizerFast"`, which makes
`transformers.AutoTokenizer.from_pretrained()` — the call `mlx_lm.tokenizer_utils.load()`
makes, i.e. what LM Studio actually runs — silently discard this model's real
pretokenizer and apply Llama's own hardcoded SentencePiece/Metaspace
conversion instead. Two keys deleted from one JSON file. No weight or model
change. Verified end-to-end with live generation.

**This document previously concluded the opposite** (single-digit
tokenization is intentional, nothing to fix). That conclusion was wrong. The
full, wrong journey is kept below — per this project's own practice — but
read this section first; nothing below it should be acted on without this
correction in mind.

## The bug, proven three ways

1. StepFun's genuinely-shipped `tokenizer.json`, loaded directly via
   `tokenizers.Tokenizer.from_file()` (no `transformers` involved at all):
   perfect round-trip on English, Chinese, and digit-heavy strings.
   `'18452'` → 2 tokens `['184','52']` — digits are **grouped** (1–3 at a
   time), not fragmented.
2. The exact same file, loaded via `AutoTokenizer.from_pretrained()` — the
   real serving path: **broken**. `'hello world'` → `'helloworld'`,
   `'你好世界'` → `''`.
3. The exact same file, wrapped in a bare `PreTrainedTokenizerFast`
   (bypassing class-name dispatch entirely): correct again.

The only variable between (2) and (3) is which Python class got
instantiated. `tokenizer_config.json` names `"LlamaTokenizerFast"` — a
**known** architecture to `transformers` — so `AutoTokenizer` applies
Llama's own conversion recipe (`Metaspace(replacement="▁", split=False)`)
instead of reading `tokenizer.json`'s actual custom 4-stage pretokenizer
Sequence. `Metaspace` expects SentencePiece-style `▁`-prefixed input; fed
raw text, it silently eats spaces and returns nothing for Chinese.

This has nothing to do with `fix_mistral_regex` (see "Superseded" below —
that flag is a real but separate false positive, and fixing it alone does
not fix the corruption).

## The fix

Delete `tokenizer_class` and `fix_mistral_regex` from `tokenizer_config.json`.
With no declared class, `AutoTokenizer` falls back to the generic
`TokenizersBackend`, which trusts `tokenizer.json`'s own pretokenizer as
authoritative — matching case (3) above.

`scripts/fix_tokenizer_class.py` does this and **verifies via a real
`AutoTokenizer` round-trip**, raising if it doesn't hold:

```bash
.venv/bin/python scripts/fix_tokenizer_class.py <checkpoint_dir>
```

Wired into every build script (`build_student{,_shared8,_tiered,_ream}.py`,
`build_head8_inplace.py`, `tomography_sweep.py`) so no future build can
silently reintroduce this.

## Verification (live generation, not just tokenizer round-trip)

LM Studio endpoint, temp 0, `top_p=1.0, top_k=0, min_p=0.0,
repetition_penalty=1.0` (all pinned so no sampler setting can be blamed or
credited):

| case | before fix | after fix |
|---|---|---|
| `kill process ID 18452` in MAC-address-dense context | 0/12 | **5/5** |
| Echo `2456, 1337, 495` in the same context (previously unresolved under every mitigation tried, including `Reasoning: medium` and Chinese-language reasoning) | 0/12 | **5/5** |

Applied to the currently-deployed checkpoint
(`~/.lmstudio/models/truemod/Step-3.7-p15-ream-shared8-head8`).

## Why this stayed hidden from everyone else

Most tooling that consumes this checkpoint doesn't route through
`transformers.AutoTokenizer`'s class-name dispatch: vLLM, llama.cpp's GGUF
converter, and any direct `tokenizers.Tokenizer.from_file()` consumer never
trigger Llama's conversion recipe. It specifically requires the
`AutoTokenizer.from_pretrained()` path, which `mlx_lm`/LM Studio uses. That
is the honest answer to "why has no one else reported this" — most
consumers simply aren't exposed to the code path that breaks it.

**StepFun's own upstream `tokenizer_config.json` carries the same
misleading `tokenizer_class` field.** This was not introduced by this
project's build pipeline. Whether it manifests for anyone else depends
entirely on how their serving stack loads the tokenizer.

## Superseded: the two earlier, wrong conclusions

Kept for the record, not for action.

**Conclusion #1 (2026-07-24), wrong: "no bug, one-digit-per-token is
intended."** This compared only two options — the deployed 2-stage
tokenizer vs. copying the base's raw 4-stage file wholesale — and never
isolated *why* the base file broke when tested. It broke because that test
*also* routed through `AutoTokenizer`'s Llama-class dispatch (compounded by
a persisted `fix_mistral_regex: true`, see below), not because StepFun's
tokenizer is inherently broken. "Intended" was never true.

**Mistake #2 (earlier the same day), compounding, now moot.** An earlier
attempt to fix this copied the base's raw `tokenizer.json` wholesale and
also carried over `fix_mistral_regex` handling, without touching
`tokenizer_class`. It partially worked (digit grouping returned) while
still routing through the same broken Llama-dispatch path for everything
else, and was reverted after it broke whitespace/CJK in a different way.

`fix_mistral_regex`'s own detection is a real, separately-filed HuggingFace
bug ([transformers#42591](https://github.com/huggingface/transformers/issues/42591)) —
it false-positives on non-Mistral models whenever `transformers_version` is
absent from `config.json`, which is exactly Step-3.7's case
(`model_type: "step3p7"`, no `transformers_version` key). That flag is a
genuine red herring for this model: fixing it alone does **not** fix the
corruption. `tokenizer_class` is the actual, confirmed, independently-tested
cause.

## Ruled out along the way (still valid negatives)

- **Quantized output head.** Digit rows carry 1.033x the all-row
  reconstruction error; perturbation is 12% of the tightest inter-digit
  argmax margin. `reap_stream/diag_head_digits.py`.
- **REAP / degraded digit copying.** Teacher-forced probes (no sampler in
  the path) rank the correct next digit first at p = 0.91–0.9996.
  `reap_stream/diag_digit_logits.py`.
- **`repetition_penalty`.** A/B at 1.02 vs 1.0: 0/8 and 1/8 vs 0/8 and 0/8.
  No effect.
- **`reasoning_effort`.** Fixed one case, made another substantially worse.
  Not a mitigation.
- **SFT / QLoRA.** Never needed — this was a two-key JSON bug.

**PPL/NLL evaluations were never confounded.** `eval_ppl_streamed.py` goes
through `transformers`, which rebuilds the tokenizer at load regardless of
which class name is declared on disk in the way that specifically breaks
serving — so the class-dispatch bug and the eval path are not the same
code path, and eval numbers in `FINDINGS.md`/`HEAD8-RESULT.md`/
`REAM-RESULT.md` are unaffected. This is exactly the instrument/deployment
gap `model-defect-triage` (Pi skill) warns about: test where it actually
runs, because the harness and the serving stack can silently disagree.

## Mitigation tooling — now defense-in-depth, not load-bearing

`scripts/numeric_guard.py` and `~/.pi/agent/extensions/numeric-guard.ts`
were built when the cause was believed intrinsic and unfixable. They can
likely be retired now that the root cause is fixed, but leaving them active
costs nothing — don't remove without re-testing first, since not every
checkpoint has had the fix applied yet.

## Still open

- CJK behavior in actual model *generation* (not just tokenizer round-trip)
  — the live verification above used English-only adversarial prompts.
- Whether checkpoints other than the currently-deployed one need the fix
  reapplied (others were deleted in an earlier disk cleanup before this fix
  was found, so only one was directly verified).
- The HF model card and the `step37-runbook` Pi skill need this correction
  propagated — check before trusting either fully.
