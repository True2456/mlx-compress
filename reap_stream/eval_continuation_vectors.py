"""Regression check against antirez/ds4's official DeepSeek-V4-Flash-0731
continuation vectors (calib/ds4_continuation_vectors/).

The vectors were captured from the real DeepSeek API with greedy decoding.
The API only exposes the actually-sampled top-1 token per step (everything
else in its top_logprobs is a -9999.0 sentinel, not a real probability), so
this is an exact-token-match regression check, not a KL/distribution
comparison. long_code_audit.txt is deliberately a repetitive, templated
"audit log" (Function f_0 / f_1 / f_2 ... near-duplicate lines) designed to
bait a model into pattern-continuation -- exactly the failure mode we're
checking for. A quantized checkpoint that starts drifting off the official
greedy path here, especially by falling into a degenerate repeat, is a
concrete, cheap signal (4 tokens, no full generation needed) worth checking
before and after any AWQ/DWQ recalibration.

Usage (run via the oMLX bundled python, or any mlx_lm-capable env that can
load the checkpoint):
    PYTHONPATH=... python -m reap_stream.eval_continuation_vectors \
        --model ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def _load_vector(vec_dir: Path, name: str) -> tuple[str, list[str]]:
    prompt = (vec_dir / f"{name}.txt").read_text()
    official = json.loads((vec_dir / f"{name}.official.json").read_text())
    expected = [s["token"]["text"] for s in official["steps"]]
    return prompt, expected


def _greedy_continue(model, tokenizer, prompt: str, n_steps: int) -> list[str]:
    from omlx.patches.deepseek_v4.chat_template_v4 import apply_chat_template

    rendered = apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        thinking_mode="chat",  # matches official vectors' "thinking": {"type": "disabled"}
    )
    ids = tokenizer.encode(rendered)
    tokens_out = []
    cache = None
    x = mx.array(ids)[None]
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    for step in range(n_steps):
        logits = model(x, cache=cache)
        next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        tokens_out.append(tokenizer.decode([next_id]))
        x = mx.array([[next_id]])
    return tokens_out


def run(model_path: str, vec_dir: str) -> None:
    from omlx.utils.model_loading import load_text_model
    from omlx.model_settings import ModelSettings

    model, tokenizer = load_text_model(model_path, model_settings=ModelSettings(mtp_enabled=False))

    vec_dir_p = Path(vec_dir)
    all_pass = True
    for name in ("long_code_audit", "long_memory_archive"):
        prompt, expected = _load_vector(vec_dir_p, name)
        got = _greedy_continue(model, tokenizer, prompt, len(expected))
        match = got == expected
        all_pass &= match
        status = "PASS" if match else "MISMATCH"
        print(f"[{status}] {name}: expected={expected} got={got}", flush=True)

    print("ALL PASS" if all_pass else "REGRESSION DETECTED", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--vec-dir",
        default=str(Path(__file__).resolve().parent.parent / "calib" / "ds4_continuation_vectors"),
    )
    a = ap.parse_args()
    run(a.model, a.vec_dir)


if __name__ == "__main__":
    main()
