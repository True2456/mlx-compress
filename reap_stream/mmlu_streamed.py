"""Likelihood-scored MMLU that streams the model block-by-block, so a model
far larger than unified memory can be measured on one machine.

Why this exists: the teacher (`DeepSeek-V4-Flash-0731`, 155GB) has never been
benchmarked, so the headroom DWQ is trying to recover is unknown. oMLX's own
MMLU is generation-based, which needs the whole model resident (or pipelined
with an uneven split across two asymmetric machines) — neither is available
here. MMLU is multiple-choice, so a single forward pass per question suffices:
compare the logits of " A"/" B"/" C"/" D" at the final position.

Absolute numbers are NOT comparable to oMLX's generation-scored MMLU (a model
can know the answer yet format its response poorly, or vice versa). Run this on
both the teacher and the student and compare the GAP — that is the quantity
that says whether DWQ has anything to recover.

Prompting is copied from omlx/eval/mmlu.py verbatim (5-shot, same subject, same
instruction line, same stratified sample) so the two models see exactly what
the real harness would show them.

Usage:
    .venv/bin/python -m reap_stream.mmlu_streamed \
        --model ~/Desktop/models/DeepSeek-V4-Flash-0731 --n 200 \
        --out artifacts/mmlu_teacher.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask

OMLX = Path("/Applications/oMLX.app/Contents/Resources")
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def _load_mmlu(n: int, seed: int = 0):
    sys.path.insert(0, str(OMLX))
    from omlx.eval.mmlu import MMLUBenchmark  # noqa: E402
    import asyncio

    b = MMLUBenchmark()
    items = asyncio.run(b.load_dataset(sample_size=n))
    return b, items


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _expand_hc(h, hc_mult):
    h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], hc_mult, h.shape[2]))
    return mx.contiguous(h)


class _DropLayer(nn.Module):
    def __call__(self, *a, **k):
        raise RuntimeError("freed layer invoked -- streaming bug")


def run(model_path: str, n: int, out_path: str, max_len: int):
    bench, items = _load_mmlu(n)
    print(f"[mmlu] {len(items)} questions", flush=True)

    # oQ / oMLX-raw checkpoints are LLM-shaped (no `language_model.` prefix);
    # our AWQ/DWQ builds are VLM-shaped. Try the VLM loader first and fall back
    # to oMLX's text loader so one script measures both.
    try:
        model, processor = load(model_path, lazy=True)
        tok = getattr(processor, "tokenizer", processor)
    except Exception as e:
        print(f"[mmlu] mlx_vlm.load failed ({type(e).__name__}); "
              "falling back to oMLX text loader", flush=True)
        sys.path.insert(0, str(OMLX))
        from omlx.utils.model_loading import load_text_model
        from omlx.model_settings import ModelSettings
        model, tok = load_text_model(model_path,
                                     model_settings=ModelSettings(mtp_enabled=False))
    lm = getattr(model, "language_model", None) or model
    text = _text_model(model)
    n_layers = len(text.layers)
    hc_mult = text.args.hc_mult
    sliding = text.args.sliding_window

    # Answer-token ids. The model predicts the token right after "Answer:",
    # which is normally " A" (leading space); fall back to bare "A".
    letter_ids = {}
    for L in "ABCD":
        cand = [tok.encode(f" {L}", add_special_tokens=False),
                tok.encode(L, add_special_tokens=False)]
        cand = [c for c in cand if c]
        letter_ids[L] = cand[0][-1] if cand else None
    print(f"[mmlu] answer token ids: {letter_ids}", flush=True)

    prompts = []
    for it in items:
        msgs = bench.format_prompt(it)
        try:
            rendered = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        except Exception:
            rendered = msgs[0]["content"]
        ids = tok.encode(rendered)
        prompts.append(ids[-max_len:] if len(ids) > max_len else ids)

    t0 = time.time()
    hidden, id_list, masks = [], [], []
    for ids in prompts:
        a = mx.array(ids)[None]
        h = _expand_hc(text.embed_tokens(a), hc_mult)
        m = create_attention_mask(h[:, :, 0, :], None, window_size=sliding,
                                  return_array=True)
        mx.eval(h, m)
        hidden.append(h); id_list.append(a); masks.append(m)

    for li in range(n_layers):
        layer = text.layers[li]
        hidden = [_run(layer, h, m, a) for h, m, a in zip(hidden, masks, id_list)]
        text.layers[li] = _DropLayer()
        gc.collect(); mx.clear_cache()
        if li % 5 == 0 or li == n_layers - 1:
            print(f"[mmlu] block {li}/{n_layers-1} "
                  f"active={mx.get_active_memory()/1e9:.1f}GB "
                  f"({time.time()-t0:.0f}s)", flush=True)

    correct = 0
    per_subject: dict[str, list[int]] = {}
    results = []
    for i, it in enumerate(items):
        collapsed = text.norm(text.hc_head(hidden[i]))
        logits = lm.lm_head(collapsed)[0][-1]
        mx.eval(logits)
        scores = {L: float(logits[tid].item()) for L, tid in letter_ids.items()
                  if tid is not None}
        pred = max(scores, key=scores.get)
        ok = pred == it["answer"]
        correct += ok
        per_subject.setdefault(it.get("subject", "?"), []).append(int(ok))
        results.append({"subject": it.get("subject"), "gold": it["answer"],
                        "pred": pred, "ok": bool(ok)})
        hidden[i] = None

    acc = correct / len(items)
    print(f"\n[mmlu] {model_path}", flush=True)
    print(f"[mmlu] accuracy: {100*acc:.1f}%  ({correct}/{len(items)})  "
          f"in {time.time()-t0:.0f}s", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "model": model_path, "n": len(items), "correct": correct,
        "accuracy": acc, "scoring": "likelihood(A/B/C/D)",
        "per_subject": {k: sum(v) / len(v) for k, v in per_subject.items()},
        "results": results,
    }, indent=2))
    print(f"[mmlu] wrote {out_path}", flush=True)


def _run(layer, h, mask, ids):
    out = layer(h, mask, None, ids)
    mx.eval(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.model, a.n, a.out, a.max_len)


if __name__ == "__main__":
    main()
