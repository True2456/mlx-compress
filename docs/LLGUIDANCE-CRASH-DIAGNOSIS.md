# LM Studio Runtime Issues: Diagnosis Log

Three separate LM Studio/Step-3.7 runtime issues diagnosed 2026-07-23, none
caused by REAP/quantization/vision-blend work. Kept in one doc since all were
found chasing the same "is this our recipe or the stack" question.

---

## 1. Structured-output crash (llguidance)

**Date:** 2026-07-23. LM Studio's generation thread was fatally crashing on
`Step-3.7-p15-4bit-vblend-shared8` with `LLGuidance matcher error: Parser
Error: token ":" doesn't satisfy the grammar`. Diagnosed and ruled out:
MTP/speculative decoding, quantization, the REAP plan, the model weights.
Root cause: a bug in `llguidance` (bundled inside LM Studio's MLX backend),
triggered by this model's tokenizer specifically. Filed upstream:
[guidance-ai/llguidance#366](https://github.com/guidance-ai/llguidance/issues/366).

## What it is NOT

- **Not MTP/speculative decoding.** `mlx_engine`'s batched-vision model kit
  (what LM Studio uses for VLMs, including step3p7) hard-refuses draft
  models: `load_draft_model()` raises "Speculative decoding is not currently
  supported for batched vision models", `is_draft_model_compatible()`
  returns `False`. There is no code path for MTP to be involved in this crash.
- **Not the model weights, quantization, or REAP plan.** Reproduced with
  zero model weights loaded -- only the tokenizer plus `llguidance.LLMatcher`
  called directly. Confirmed on both the local checkpoint and a fresh
  download of the public tokenizer (`stepfun-ai/Step-3.7-Flash`), same
  failure both times.
- **Not something we can patch locally.** The bug is inside llguidance's
  Rust core, vendored inside LM Studio's `mlx-generate` backend extension.

## What it IS

LM Studio's auto chat-title generation issues a JSON-schema grammar
(`{"title": string, minLength: 1}`) with `x-guidance.key_separator: ": "`
(colon + space, two bytes). Step-3.7's tokenizer has a merged token that
decodes to `":` (quote immediately followed by colon). Driving the grammar
greedily (always consuming whichever token `fill_next_token_bitmask()` marks
as the sole legal choice) shows the mask generator and the token consumer
contradict each other:

```
step 2: about to consume ':' (id=28) | allowed_by_mask=True | n_allowed_tokens=1
  -> MATCHER ERROR: Parser Error: token ":" doesn't satisfy the grammar;
     forced bytes: got ' '; applying ':'
```

The bitmask says `:` is the *only* legal token at this step; consuming that
exact token is then rejected. This is a genuine llguidance bug, not a usage
error -- the caller followed the mask exactly.

**Isolated trigger, confirmed by variant testing:**

| variant | tokenizer | result |
|---|---|---|
| `key_separator: ": "` (LM Studio's actual schema) | step3p7 | **fails** |
| `key_separator: ":"` (no trailing space) | step3p7 | works |
| no `x-guidance` block at all | step3p7 | works |
| `key_separator: ": "` (identical schema) | Gemma-4 | **works** |

So it's specifically: multi-byte forced separator + this tokenizer's
merged `":` token. Not reproducible on a different tokenizer with the
identical grammar, and not reproducible on this tokenizer with a
single-byte separator.

## Practical workaround

Disable LM Studio's automatic chat-title generation (the feature issuing
this exact schema). Normal chat generation is unaffected -- this only fires
on structured-output/JSON-schema requests against this model. Any other
JSON-schema call with a multi-character key/item separator will hit the same
wall until llguidance ships a fix.

## Bonus finding: MTP weights exist but were dropped by our build

Investigating the (ruled-out) MTP theory surfaced something worth recording.
The BF16 original (`models/Step-3.7-Flash`) DOES contain MTP weights --
DeepSeek-style, three extra decoder-indexed layers (45, 46, 47) with
`eh_proj`, `enorm`, `hnorm`, `transformer.shared_head.{norm,output}`. Every
quantized model built this session (`vblend`, `vblend-shared8`, the
tomography variants) silently dropped them, because `mlx_vlm`'s step3p7
implementation only models layers 0-44 -- `build_student.py`'s REAP-apply
pass never touches or copies layers beyond that range.

This is not a bug in anything we built (LM Studio has no code path to use
MTP for this model regardless -- see above), just a fact worth knowing: the
MTP weights are sitting unused in `models/Step-3.7-Flash`, identical in kind
to the ones published separately as
[Hikari07jp/Step-3.7-Flash-MTP-draft](https://huggingface.co/Hikari07jp/Step-3.7-Flash-MTP-draft)
(that repo targets vLLM's `Step3p5MTP`/`Step3p5MTPProposer` via
`--speculative-config`, not MLX). If MTP ever becomes usable on this stack,
the weights don't need to be re-downloaded -- extracting them from the local
BF16 checkpoint is sufficient. Making them actually usable would require an
MLX-side implementation of the MTP proposer logic (feed the accepted token's
hidden state through `eh_proj` + the small extra decoder block to propose the
next token) -- `mlx_vlm` does have generic draft-model speculative-decoding
plumbing (`server/generation.py`, `sample_utils.py`, `utils.py`), but that's
architecture-agnostic draft-model speculation, not an MTP proposer
specifically, and using it for VLMs is unverified. A real project, not a
config change; not attempted here.

### Artifacts

Model-free repro scripts (verified against both the local checkpoint and the
public `stepfun-ai/Step-3.7-Flash` tokenizer): see the issue link above for
the canonical version. Filed upstream at
[guidance-ai/llguidance#366](https://github.com/guidance-ai/llguidance/issues/366).

---

## 2. Segfault sending a video file as an "image"

**Symptom:** "The model has crashed without additional information. (Exit
code: null)" after attaching a file and asking the model to describe it.

**Diagnosis:** the attached file, `Screen Recording 2026-07-17 at 3.23.04
pm.mov` (`~/.lmstudio/user-files/`), is a QuickTime video container --
confirmed via `file`, and LM Studio's own attachment metadata records
`"type": "unknown"`. It was routed into the vision pipeline anyway (RAG
retrieval query: `"describe this image analytically"`). LM Studio's
server log (`~/.lmstudio/server-logs/2026-07/2026-07-23.1.log`) shows the
timeline:

```
19:10:44  RAG retrieval requested against the .mov file
19:10:51  VLM prompt cache write (image tokens encoded into context)
19:11:10  Fatal Python error: Segmentation fault
```

19 seconds between the vision encoder accepting the file and a hard
segfault. The crash's Python thread dump shows only idle worker threads
waiting on queues (`<no Python frame>` on the actual faulting thread) --
the fault happened in native code (Metal/MLX C++), which is exactly why
LM Studio could report no further detail. Consistent with a malformed/
mismatched tensor shape (video-container bytes decoded as if a static
image) reaching a native kernel. Not an OOM -- no jetsam/swap events
around the timestamp.

**Fix:** don't attach video files as images. Export a still frame (e.g.
QuickTime Player -> step to frame -> Edit -> Copy, or a fresh screenshot)
and attach that instead. The model has no video/temporal understanding
regardless, so a still frame is the correct input independent of the crash.

---

## 3. Jinja template error: "Unknown StringValue filter: safe"

**Symptom:** `Error rendering prompt with jinja template: "Unknown
StringValue filter: safe"` on some tool-calling turns.

**Diagnosis:** `chat_template.jinja` (shipped with the original
`stepfun-ai/Step-3.7-Flash` release, copied verbatim into every build by
`build_student.py`/`build_student_shared8.py`) has, in its tool-call
argument re-serialization block:

```jinja
{%- set args_value = args_value | tojson(ensure_ascii=False) | safe if ... %}
```

`| safe` is a Flask/Jinja2 idiom that suppresses HTML-escaping -- meaningless
in a plain-text chat template, and a no-op on the actual rendered output. But
LM Studio bundles `minijinja` (a Rust Jinja2 implementation) rather than
Python Jinja2, and `minijinja` does not implement a `safe` filter, so it
hard-fails instead of ignoring it. Triggers specifically when a tool-call
argument is a `dict`/`list` (not a plain string) -- i.e. any nontrivial
tool_use turn, which is common given this model's heavy tool-calling
training mix.

**Fix applied:** removed `| safe` from `chat_template.jinja` in the BF16
source (`models/Step-3.7-Flash/`, so all future builds inherit the fix) and
in both currently-deployed models
(`~/.lmstudio/models/truemod/Step-3.7-p15-4bit-vblend/` and
`.../Step-3.7-p15-4bit-vblend-shared8/`). Pure no-op removal -- does not
change any rendered output, only stops `minijinja` from refusing to render.
Not filed upstream (StepFun's repo, not a tooling dependency with an
obvious issue tracker fit like llguidance) -- fixed locally instead.
