"""Render cloud_reap_8k into Qwen3.5 chat-template format for calibration.

Why this exists: AWQ's scale search keys on the activations a real prompt
produces, so the calibration text must carry the same STRUCTURAL tokens the
model sees at inference. The v1 -> v2 DeepSeek jump (MMLU 34.5% -> 61.5%) came
almost entirely from fixing exactly this -- v1 calibrated on single-turn text
for a model deployed on multi-turn agentic conversations.

cloud_reap_8k stores plain `SYSTEM:/USER:/ASSISTANT:/TOOLS:/TOOL:` markers,
which are NOT what Qwen sees. Qwen wraps turns in <|im_start|>, may inject a
reasoning_effort instruction, and opens a <think> block. So each record is
parsed back into messages and re-rendered through the model's own template.

think/nothink is split 50/50, mirroring the ds4_agentic corpus that produced
the v2 result (2345/2345), so the calibration covers both deployment modes.

Multimodal rows are excluded: their text ("When does the positive view reach
the peak?") is degenerate without the chart, and the vision tower is not part
of the AWQ scope.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MARKERS = ("SYSTEM:", "USER:", "ASSISTANT:", "TOOLS:", "TOOL:")
_SPLIT = re.compile(r"^(SYSTEM:|USER:|ASSISTANT:|TOOLS:|TOOL:)\s*$", re.M)
_ROLE = {"SYSTEM:": "system", "USER:": "user", "ASSISTANT:": "assistant", "TOOL:": "tool"}


def parse_turns(text: str):
    """-> (messages, tools) or (None, None) if the record isn't parseable."""
    parts = _SPLIT.split(text)
    if len(parts) < 3:
        return None, None
    msgs, tools = [], None
    # parts[0] is any preamble before the first marker; ignore if blank
    it = iter(range(1, len(parts) - 1, 2))
    for i in it:
        marker, body = parts[i], parts[i + 1].strip()
        if marker == "TOOLS:":
            try:
                t = json.loads(body)
                tools = t if isinstance(t, list) else None
            except Exception:
                tools = None
            continue
        role = _ROLE[marker]
        if not body:
            continue
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + body
        else:
            msgs.append({"role": role, "content": body})
    return (msgs or None), tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/true/Desktop/models/Qwen3.8-27B")
    ap.add_argument("--dataset", default="calib/cloud_reap_8k.jsonl")
    ap.add_argument("--out", default="calib/qwen38_calib.jsonl")
    ap.add_argument("--exclude-categories", nargs="*", default=["multimodal"])
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="reporting only -- records are not truncated here")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    rows = [json.loads(l) for l in open(a.dataset) if l.strip()]
    excl = set(a.exclude_categories)
    kept, skipped, lens = [], {"category": 0, "unparsed": 0, "template": 0}, []

    for r in rows:
        if r.get("category") in excl:
            skipped["category"] += 1
            continue
        msgs, tools = parse_turns(r.get("text", ""))
        if not msgs or not any(m["role"] == "user" for m in msgs):
            skipped["unparsed"] += 1
            continue
        # template requires system first; drop stray later system turns
        msgs = [m for i, m in enumerate(msgs) if m["role"] != "system" or i == 0]
        # 50/50 think / nothink, deterministic by index
        think = (len(kept) % 2 == 0)
        try:
            kw = dict(tokenize=False, add_generation_prompt=False)
            if tools:
                kw["tools"] = tools
            if think:
                kw["enable_thinking"] = True
                kw["preserve_thinking"] = False   # history without empty <think>
            else:
                kw["enable_thinking"] = False
            text = tok.apply_chat_template(msgs, **kw)
        except Exception:
            skipped["template"] += 1
            continue
        n = len(tok.encode(text))
        lens.append(n)
        kept.append({
            "text": text, "prerendered": True,
            "category": r.get("category", "?"),
            "mode": "think" if think else "nothink",
            "n_tokens": n,
        })

    out = Path(a.out)
    with out.open("w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")

    import statistics as st
    lens.sort()
    def pct(p): return lens[min(len(lens) - 1, int(p / 100 * len(lens)))]
    print(f"[calib] wrote {out}  kept={len(kept)}  skipped={skipped}")
    print(f"[calib] token lengths: p10={pct(10)} p50={pct(50)} p90={pct(90)} "
          f"p99={pct(99)} max={lens[-1]}")
    for cap in (384, 512, 768, 1024, 1536, 2048):
        trunc = sum(1 for n in lens if n > cap)
        real = sum(min(n, cap) for n in lens)
        print(f"   cap {cap:5d}: truncated {100*trunc/len(lens):5.1f}%  "
              f"padding {100*(1-real/(len(lens)*cap)):5.1f}%")


if __name__ == "__main__":
    main()
