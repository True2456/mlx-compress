"""Long-horizon repetition/looping probe.

The ds4 continuation vectors (eval_continuation_vectors.py) only check 4
greedy tokens -- too short to catch the looping behavior observed in real
extended agentic use, which shows up tens to hundreds of tokens into a
generation. This script instead:

1. Loads a handful of genuine multi-turn agentic trajectories (real
   system/user/assistant/tool sequences, see convert_ds4_imatrix_prompts.py
   for provenance) from calib/ds4_agentic_repetition_probes.jsonl.
2. Drops each trajectory's final assistant turn and re-renders the prefix
   with add_generation_prompt=True, so our model has to generate its own
   continuation from a realistic multi-step agent context -- not replay a
   cached teacher continuation.
3. Greedily generates up to --max-tokens fresh tokens and scores the output
   for degenerate repetition:
     - distinct-n ratio (unique n-grams / total n-grams) for n=3,4 -- the
       standard text-generation repetition metric (Holtzman et al.); lower
       means more repetitive.
     - longest verbatim repeated token span -- directly catches the
       "generates identical block over and over" failure mode, which
       distinct-n alone can under-report if the repeated block is long
       relative to the whole generation.

Usage:
    PYTHONPATH=... python -m reap_stream.eval_repetition \
        --model ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw \
        --max-tokens 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def _distinct_n(tokens: list[int], n: int) -> float:
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def _longest_verbatim_repeat(tokens: list[int], min_len: int = 6) -> int:
    """Longest span length L such that some subsequence of length L appears
    at least twice (non-overlapping) in tokens. O(n^2) is fine at ~300-400
    tokens."""
    n = len(tokens)
    best = 0
    for start in range(n):
        for length in range(min_len, n - start + 1):
            span = tokens[start : start + length]
            rest = tokens[start + length :]
            found = False
            for j in range(len(rest) - length + 1):
                if rest[j : j + length] == span:
                    found = True
                    break
            if found:
                best = max(best, length)
            else:
                break
    return best


def _greedy_generate(model, tokenizer, prompt_ids: list[int], max_tokens: int) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    x = mx.array(prompt_ids)[None]
    out_ids = []
    eos_ids = set(tokenizer.eos_token_ids) if hasattr(tokenizer, "eos_token_ids") else {tokenizer.eos_token_id}
    for _ in range(max_tokens):
        logits = model(x, cache=cache)
        next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if next_id in eos_ids:
            break
        out_ids.append(next_id)
        x = mx.array([[next_id]])
    return out_ids


def run(model_path: str, probes_path: str, max_tokens: int) -> None:
    from omlx.utils.model_loading import load_text_model
    from omlx.model_settings import ModelSettings
    from omlx.patches.deepseek_v4.chat_template_v4 import apply_chat_template

    model, tokenizer = load_text_model(model_path, model_settings=ModelSettings(mtp_enabled=False))

    flagged = 0
    total = 0
    with open(probes_path) as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec["messages"]
            assert msgs[-1]["role"] == "assistant"
            prefix_msgs = msgs[:-1]

            rendered = apply_chat_template(prefix_msgs, add_generation_prompt=True, thinking_mode="chat")
            prompt_ids = tokenizer.encode(rendered)

            gen_ids = _greedy_generate(model, tokenizer, prompt_ids, max_tokens)
            text = tokenizer.decode(gen_ids)

            d3 = _distinct_n(gen_ids, 3)
            d4 = _distinct_n(gen_ids, 4)
            longest_repeat = _longest_verbatim_repeat(gen_ids)

            # Heuristic flags: distinct-4 well below natural-text baseline
            # (~0.85-0.95 for coherent English/code), or a long verbatim
            # repeated block (looping, not just natural short phrase reuse).
            is_flagged = d4 < 0.5 or longest_repeat >= 20
            total += 1
            flagged += is_flagged

            status = "LOOP-SUSPECT" if is_flagged else "ok"
            print(
                f"[{status}] {rec['id']} ({rec['source']}): "
                f"n_gen={len(gen_ids)} distinct-3={d3:.3f} distinct-4={d4:.3f} "
                f"longest_verbatim_repeat={longest_repeat}",
                flush=True,
            )
            if is_flagged:
                print(f"    tail: {text[-300:]!r}", flush=True)

    print(f"\n{flagged}/{total} probes flagged as loop-suspect", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--probes",
        default=str(Path(__file__).resolve().parent.parent / "calib" / "ds4_agentic_repetition_probes.jsonl"),
    )
    ap.add_argument("--max-tokens", type=int, default=300)
    a = ap.parse_args()
    run(a.model, a.probes, a.max_tokens)


if __name__ == "__main__":
    main()
