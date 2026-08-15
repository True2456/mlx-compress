"""Measure the DSpark drafter's accept rate directly, isolated from the
serving engine.

Rationale: DSpark verifies every drafted token against the main model with
exact rejection sampling, so quantizing the drafter can only cost ACCEPT RATE
(speed), never correctness. Accept rate is therefore the single number that
decides whether an MTP quantization is worth shipping -- and unlike end-to-end
tokens/sec it doesn't depend on engine plumbing, batch settings or thermal
state, so it compares cleanly across checkpoints.

Protocol, per prompt, under greedy decoding:
  1. Run the main model over the prompt -> its greedy next token, plus the
     DSpark target-layer taps (``h_aux``) that condition the drafter.
  2. Ask DSpark for a block of ``--depth`` draft tokens from those taps.
  3. Advance the main model one token at a time to get the true greedy
     continuation, and compare position-wise.

Position j is only counted once every earlier position matched, mirroring the
real accept semantics (a rejected draft discards the rest of the block).

Usage:
    PYTHONPATH=... python -m reap_stream.eval_mtp_accept_rate \
        --model ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw-v2 \
        --depth 4 --n-prompts 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def _prompts(path: Path, n: int) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text")
            if text and text.strip():
                out.append(text.strip())
            if len(out) >= n:
                break
    return out


def run(model_path: str, prompts_path: str, n_prompts: int, depth: int, max_ctx: int) -> None:
    from omlx.model_settings import ModelSettings
    from omlx.utils.model_loading import load_text_model

    model, tok = load_text_model(
        model_path, model_settings=ModelSettings(mtp_enabled=True, mtp_num_draft_tokens=depth)
    )
    if not getattr(model, "_omlx_dspark_decode_enabled", False):
        raise SystemExit("DSpark decode not enabled on this model -- cannot measure accept rate")

    drafted = [0] * depth
    accepted = [0] * depth
    n_used = 0

    for text in _prompts(Path(prompts_path), n_prompts):
        ids = tok.encode(text)[:max_ctx]
        if len(ids) < 8:
            continue
        inputs = mx.array(ids)[None]

        main_cache = model.make_cache()
        logits, h_aux = model(inputs, cache=main_cache, return_hidden=True)
        mx.eval(logits, h_aux)

        # DSpark's proposal block, conditioned on the prompt's target taps.
        mtp_cache = model.make_mtp_cache()
        draft_logits, _ = model.dspark_forward(
            h_aux, inputs, cache=mtp_cache, draft_length=depth
        )
        mx.eval(draft_logits)
        drafts = [int(mx.argmax(draft_logits[0, j]).item()) for j in range(draft_logits.shape[1])]

        # True greedy continuation from the main model.
        truth: list[int] = []
        nxt = int(mx.argmax(logits[0, -1]).item())
        truth.append(nxt)
        for _ in range(len(drafts) - 1):
            step_logits = model(mx.array([[nxt]]), cache=main_cache)
            mx.eval(step_logits)
            nxt = int(mx.argmax(step_logits[0, -1]).item())
            truth.append(nxt)

        alive = True
        for j in range(min(len(drafts), len(truth), depth)):
            if not alive:
                break
            drafted[j] += 1
            if drafts[j] == truth[j]:
                accepted[j] += 1
            else:
                alive = False
        n_used += 1
        mx.clear_cache()

    print(f"\nmodel: {model_path}")
    print(f"prompts: {n_used}, depth: {depth}")
    for j in range(depth):
        rate = accepted[j] / drafted[j] if drafted[j] else float("nan")
        print(f"  depth {j+1}: {accepted[j]}/{drafted[j]} = {rate:.3f}")
    # Expected tokens emitted per verify cycle under greedy chained acceptance.
    exp = 0.0
    surv = 1.0
    for j in range(depth):
        r = accepted[j] / drafted[j] if drafted[j] else 0.0
        surv *= r
        exp += surv
    print(f"  expected accepted drafts per cycle: {exp:.3f} (+1 verified token)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="calib/ds4_agentic.jsonl")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-ctx", type=int, default=768)
    a = ap.parse_args()
    run(a.model, a.prompts, a.n_prompts, a.depth, a.max_ctx)


if __name__ == "__main__":
    main()
