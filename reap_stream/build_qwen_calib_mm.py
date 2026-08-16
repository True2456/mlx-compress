"""Build a Qwen3.5 calibration set that includes real images.

The text-only calibration used for the first build never ran the vision tower,
so its 167 modules had no imatrix data and their 8-bit assignment was a guess
rather than a measurement (see docs/QWEN38-FINDINGS.md S3). It also meant the
language model was calibrated without ever seeing image tokens, which for a VLM
is off-distribution for a real share of its traffic.

Sources, all rendered through Qwen's own chat template:

  cloud_reap_8k text          coding / tool_use / agentic / reasoning
  cloud_reap_8k multimodal    ChartQA-style rows, images on disk
  iq-terrain-vlm-dataset      GLSL / raymarching / shader renders
  multimodal_mixed (Step-3.7) ChartQA + vqav2-small natural photographs
  multimodal_fixed (Step-3.7) ChartQA

calib/multimodal_eval/ is deliberately excluded: it carries prefix/answer
fields and is the evaluation set, so calibrating on it would contaminate it.

Rows carrying an image emit `image_path` alongside `text`; the AWQ driver feeds
those through the processor so the vision tower actually executes and its
activations enter the language model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reap_stream.build_qwen_calib import parse_turns


def _render(tok, msgs, think: bool, tools=None):
    kw = dict(tokenize=False, add_generation_prompt=False)
    if tools:
        kw["tools"] = tools
    if think:
        kw["enable_thinking"] = True
        kw["preserve_thinking"] = False
    else:
        kw["enable_thinking"] = False
    return tok.apply_chat_template(msgs, **kw)


def _iq_terrain(path: str, tok, limit: int, out: list, stats: dict):
    """messages + images, with a literal `<image>` placeholder in the user turn.

    Qwen expects its own vision sentinels, which apply_chat_template inserts
    when the content is a list containing an image part. So the placeholder is
    stripped and the content re-expressed in Qwen's structured form.
    """
    if not os.path.exists(path):
        stats["iq_missing"] = path
        return
    for i, line in enumerate(open(path)):
        if not line.strip() or len([r for r in out if r.get("source") == "iq"]) >= limit:
            continue
        rec = json.loads(line)
        imgs = rec.get("images") or []
        if not imgs or not os.path.exists(imgs[0]):
            stats["iq_no_image"] = stats.get("iq_no_image", 0) + 1
            continue
        msgs = []
        for m in rec["messages"]:
            c = m["content"]
            if m["role"] == "user" and "<image>" in c:
                c = c.replace("<image>", "").strip()
                msgs.append({"role": "user",
                             "content": [{"type": "image"}, {"type": "text", "text": c}]})
            else:
                msgs.append({"role": m["role"], "content": c})
        try:
            text = _render(tok, msgs, think=(i % 2 == 0))
        except Exception:
            stats["iq_template_fail"] = stats.get("iq_template_fail", 0) + 1
            continue
        out.append({"text": text, "image_path": imgs[0], "category": "shader_vlm",
                    "source": "iq", "prerendered": True})


def _step37_mm(path: str, tok, limit: int, out: list, stats: dict, tag: str):
    """Step-3.7-era multimodal sets: `<im_patch>` placeholder + USER:/ASSISTANT:.

    multimodal_mixed carries 100 natural-image VQA rows (merve/vqav2-small)
    alongside the charts; nothing else in this calibration has photographs.
    NOTE: calib/multimodal_eval/ is deliberately NOT read here -- it has
    prefix/answer fields and is the evaluation set.
    """
    if not os.path.exists(path):
        stats[f"{tag}_missing"] = path
        return
    n = 0
    for i, line in enumerate(open(path)):
        if not line.strip() or n >= limit:
            continue
        rec = json.loads(line)
        img = rec.get("image")
        if not img or not os.path.exists(img):
            stats[f"{tag}_no_image"] = stats.get(f"{tag}_no_image", 0) + 1
            continue
        body = rec.get("text", "").replace("<im_patch>", "").strip()
        msgs, _ = parse_turns(body)
        if not msgs or not any(m["role"] == "user" for m in msgs):
            stats[f"{tag}_unparsed"] = stats.get(f"{tag}_unparsed", 0) + 1
            continue
        for m in msgs:
            if m["role"] == "user":
                m["content"] = [{"type": "image"}, {"type": "text", "text": m["content"]}]
                break
        try:
            text = _render(tok, msgs, think=(i % 2 == 0))
        except Exception:
            stats[f"{tag}_template_fail"] = stats.get(f"{tag}_template_fail", 0) + 1
            continue
        out.append({"text": text, "image_path": img, "source": tag,
                    "category": rec.get("subtype") or "chart", "prerendered": True})
        n += 1


def _cloud(path: str, tok, text_limit: int, mm_limit: int, out: list, stats: dict):
    n_text = n_mm = 0
    for i, line in enumerate(open(path)):
        if not line.strip():
            continue
        rec = json.loads(line)
        img = rec.get("image")
        is_mm = bool(img)
        if is_mm and n_mm >= mm_limit:
            continue
        if not is_mm and n_text >= text_limit:
            continue
        msgs, tools = parse_turns(rec.get("text", ""))
        if not msgs or not any(m["role"] == "user" for m in msgs):
            stats["cloud_unparsed"] = stats.get("cloud_unparsed", 0) + 1
            continue
        msgs = [m for j, m in enumerate(msgs) if m["role"] != "system" or j == 0]
        if is_mm:
            p = img if os.path.isabs(img) else str(Path(img))
            if not os.path.exists(p):
                stats["cloud_no_image"] = stats.get("cloud_no_image", 0) + 1
                continue
            for m in msgs:
                if m["role"] == "user":
                    m["content"] = [{"type": "image"},
                                    {"type": "text", "text": m["content"]}]
                    break
        try:
            text = _render(tok, msgs, think=(i % 2 == 0), tools=tools)
        except Exception:
            stats["cloud_template_fail"] = stats.get("cloud_template_fail", 0) + 1
            continue
        row = {"text": text,
               "category": "chart" if is_mm else rec.get("category", "?"),
               "source": "cloud", "prerendered": True}
        if is_mm:
            row["image_path"] = p
            n_mm += 1
        else:
            n_text += 1
        out.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/true/Desktop/models/Qwen3.8-27B")
    ap.add_argument("--cloud", default="calib/cloud_reap_8k.jsonl")
    ap.add_argument("--iq", default="/tmp/iq-terrain/gemma4/train.jsonl")
    ap.add_argument("--out", default="calib/qwen38_calib_mm.jsonl")
    ap.add_argument("--text-rows", type=int, default=1200)
    ap.add_argument("--cloud-mm-rows", type=int, default=250)
    ap.add_argument("--iq-rows", type=int, default=250)
    ap.add_argument("--mixed", default="calib/multimodal_mixed/multimodal_mixed.jsonl")
    ap.add_argument("--fixed", default="calib/multimodal_fixed/multimodal_fixed.jsonl")
    ap.add_argument("--mixed-rows", type=int, default=300)  # chart-first file; 300 reaches the natural-image tail
    ap.add_argument("--fixed-rows", type=int, default=100)
    ap.add_argument("--pct-natural", type=int, default=10)
    ap.add_argument("--pct-chart", type=int, default=10)
    ap.add_argument("--pct-shader", type=int, default=10)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    out: list = []
    stats: dict = {}
    _cloud(a.cloud, tok, a.text_rows, a.cloud_mm_rows, out, stats)
    _iq_terrain(a.iq, tok, a.iq_rows, out, stats)
    _step37_mm(a.mixed, tok, a.mixed_rows, out, stats, "mixed")
    _step37_mm(a.fixed, tok, a.fixed_rows, out, stats, "fixed")

    # Stratify to a target image ratio, and rotate image DOMAINS so that any
    # prefix of the file has the intended mix. A naive round-robin over sources
    # put 80% images in the first 352 rows and never reached the natural-image
    # tail -- the model is used mostly for text, and the text benchmarks are
    # what we measure, so images must not dominate.
    import random
    rng = random.Random(0)
    texts = [r for r in out if not r.get("image_path")]
    imgs = [r for r in out if r.get("image_path")]
    rng.shuffle(texts)
    by_dom: dict = {}
    for r in imgs:
        by_dom.setdefault(r.get("category", "?"), []).append(r)
    for v in by_dom.values():
        rng.shuffle(v)
    # explicit per-domain share of the whole file, held across any prefix
    quota = {"natural": a.pct_natural, "chart": a.pct_chart,
             "shader_vlm": a.pct_shader}
    doms = [d for d in ("natural", "chart", "shader_vlm") if by_dom.get(d)]
    for d in list(by_dom):
        if d not in quota:
            doms.append(d)
            quota[d] = 0
    inter, di, ti, k = [], 0, 0, 0
    while ti < len(texts) or any(by_dom[d] for d in doms):
        want_image = (k % 100) < sum(quota.values())
        placed = False
        if want_image:
            # pick the domain furthest below its quota so shares hold early
            target = {d: quota[d] / 100.0 for d in doms}
            have = {d: sum(1 for r in inter if r.get("category") == d) for d in doms}
            order = sorted(doms, key=lambda d: (have[d] / max(k, 1)) - target[d])
            for d in order:
                if by_dom[d] and quota[d] > 0:
                    inter.append(by_dom[d].pop()); placed = True; break
        if not placed and ti < len(texts):
            inter.append(texts[ti]); ti += 1; placed = True
        if not placed:
            for d in doms:
                if by_dom[d]:
                    inter.append(by_dom[d].pop()); placed = True; break
        if not placed:
            break
        k += 1

    with open(a.out, "w") as f:
        for r in inter:
            f.write(json.dumps(r) + "\n")

    n_img = sum(1 for r in inter if r.get("image_path"))
    lens = sorted(len(tok.encode(r["text"])) for r in inter)
    print(f"[calib-mm] wrote {a.out}: {len(inter)} rows, {n_img} with images "
          f"({100 * n_img / max(len(inter), 1):.0f}%)")
    print(f"[calib-mm] skipped: {stats or 'none'}")
    print(f"[calib-mm] tokens p50={lens[len(lens)//2]} p90={lens[int(.9*len(lens))]} "
          f"max={lens[-1]}")
    for cap in (768, 1024, 1536):
        tr = sum(1 for n in lens if n > cap)
        print(f"   cap {cap}: truncated {100*tr/len(lens):.0f}%  "
              f"padding {100*(1-sum(min(n,cap) for n in lens)/(len(lens)*cap)):.0f}%")


if __name__ == "__main__":
    main()
