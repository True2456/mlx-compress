"""Did REAP/quantization break in-context digit copying? Teacher-forced probe.

Measured symptom (LM Studio, temp 0.0, greedy -- so this is a logit fault, not
a sampling one): in a MAC-address-dense context, "kill process 18452" is
answered `kill -9 1845` on 5/5 samples. The trailing digit is dropped. Same
prompt with no context is clean.

`diag_head_digits.py` already exonerated the output head (digit rows carry
1.03x the all-row quant error; perturbation is 12% of the tightest inter-digit
margin). That says nothing about the 42 pruned MoE layers feeding it, and
exact digit copying is mid-stack induction behaviour -- precisely what expert
pruning could damage while leaving aggregate PPL flat, since a 500-prompt
average cannot see a copy-accuracy fault.

No unpruned Step-3.7 exists as a loadable quant (the 148B on the Hub is a
*different* REAP at 212/288 experts and nvfp4, so it is not a control). The
only true reference is the BF16 base, too large to load -- but this probe does
not need generation. Force a prefix ending one token short of the digit in
question and read the next-token distribution, streaming decoder blocks for
BF16 exactly as eval_ppl_streamed.py does.

If BF16 ranks the correct digit first and the student does not, the student
is damaged. If BF16 also misses, the fault is inherited from upstream.

Usage:
    # student (resident, fast)
    .venv/bin/python -m reap_stream.diag_digit_logits \
        --model models/Step-3.7-p15-4bit-vblend-shared8

    # BF16 reference (streamed, slow, ~34GB peak)
    .venv/bin/python -m reap_stream.diag_digit_logits \
        --model models/Step-3.7-Flash --stream
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.utils import fetch_from_hub, get_model_path

from .collect_step3p7 import _free_layer, _run_layer, _text_config, _text_model

MAC_CTX = """```
# arp -a
gateway   (192.168.1.1)  at 2a:5b:c4:9f:1d:e0 on en0
printer   (192.168.1.24) at 3c:df:a9:b2:44:71 on en0
nas       (192.168.1.31) at 8e:1f:0c:5a:d7:b3 on en0
laptop    (192.168.1.47) at f0:18:98:6e:2b:cc on en0
sensor    (192.168.1.62) at 04:e9:e5:1a:9f:38 on en0
camera    (192.168.1.88) at b8:27:eb:7c:03:5d on en0
switch    (192.168.1.2)  at 6c:3b:6b:d1:8f:04 on en0
ap        (192.168.1.3)  at de:ad:be:ef:00:11 on en0
```"""

# (label, user turn, forced assistant prefix, expected next token)
PROBES = [
    ("kill_hex",   MAC_CTX + "\n\nReply with ONLY the shell command to kill "
                             "process ID 18452 with signal 9.",
     "kill -9 1845", "2"),
    ("kill_plain", "Reply with ONLY the shell command to kill process ID 18452 "
                   "with signal 9.",
     "kill -9 1845", "2"),
    ("echo_hex",   MAC_CTX + "\n\nReply with ONLY these numbers separated by "
                             "commas, exactly as written: 2456, 1337, 495",
     "2456, 1337, 49", "5"),
    ("echo_plain", "Reply with ONLY these numbers separated by commas, exactly "
                   "as written: 2456, 1337, 495",
     "2456, 1337, 49", "5"),
]


def build_ids(processor, user_text, forced):
    """Chat-templated prompt + closed think block + forced answer prefix.

    The template pre-opens <think>, so the generation prompt already ends
    inside the reasoning block; close it immediately to make the two models
    directly comparable without depending on identical chains of thought.
    """
    tok = getattr(processor, "tokenizer", processor)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True, tokenize=False,
    )
    return tok, tok.encode(prompt + "</think>\n\n" + forced,
                           add_special_tokens=False)


def logits_resident(model, ids):
    out = model.language_model(mx.array(ids)[None])
    return (out.logits if hasattr(out, "logits") else out)[0, -1]


def logits_streamed(model, ids):
    text, lm = _text_model(model), model.language_model
    sliding = getattr(_text_config(model), "sliding_window", None)
    h = text.embed_tokens(mx.array(ids)[None])
    mx.eval(h)
    for li in range(len(text.layers)):
        h = _run_layer(text.layers[li], h, sliding)
        mx.eval(h)
        _free_layer(text, li)
        if li % 8 == 0:
            gc.collect()
            mx.clear_cache()
            print(f"    layer {li}/{len(text.layers)-1} "
                  f"active={mx.get_active_memory()/1e9:.1f}GB", flush=True)
    return lm.lm_head(text.norm(h))[0, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stream", action="store_true",
                    help="stream decoder blocks (BF16 reference)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.stream:
        src = get_model_path(a.model)
        model, _, processor = fetch_from_hub(src, lazy=True, trust_remote_code=True)
    else:
        model, processor = load(a.model, trust_remote_code=True)

    report = {}
    for label, user_text, forced, expect in PROBES:
        tok, ids = build_ids(processor, user_text, forced)
        lg = (logits_streamed if a.stream else logits_resident)(model, ids)
        lg = lg.astype(mx.float32)
        probs = mx.softmax(lg)
        order = mx.argsort(-lg)[: a.topk].tolist()
        want = tok.encode(expect, add_special_tokens=False)[0]
        rank = int(mx.sum(lg > lg[want]))
        top = [(tok.decode([i]), round(float(probs[i]), 4)) for i in order]
        ok = "OK " if rank == 0 else "MISS"
        print(f"\n[{ok}] {label}: forced {forced!r} -> want {expect!r}")
        print(f"       want rank={rank}  p={float(probs[want]):.4f}")
        print(f"       top{a.topk}: {top}")
        report[label] = {"expect": expect, "rank": rank,
                         "p_expect": float(probs[want]), "top": top}
        if a.stream:      # streaming consumed the layers; reload for next probe
            src = get_model_path(a.model)
            model, _, processor = fetch_from_hub(src, lazy=True,
                                                 trust_remote_code=True)
            gc.collect()
            mx.clear_cache()

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\n[INFO] wrote {a.out}")


if __name__ == "__main__":
    main()
