"""Build train/valid/test.jsonl for mlx_lm.lora from the greghavens traces.

Combines both greghavens SFT trace datasets into mlx_lm's native ChatDataset
format ({"messages": [...], "tools": [...]}). Key transforms, each with a
reason:

  * Parse tool_call `arguments` from JSON string -> dict. Step-3.7's chat
    template calls `arguments | fromjson` only when arguments is a string;
    HF's apply_chat_template has no `fromjson` filter, so string-argument
    rows fail to render. Pre-parsing to dict takes the template's else
    branch and renders cleanly (verified: fixes 100% of Fable-5 failures).
  * Parse the `tools` JSON-string field -> list.
  * Map teacher reasoning_effort -> Step-3.7's three levels
    (max/xhigh -> high, medium -> medium, low -> low) and inject
    "Reasoning: <level>\n\n" at the front of the system message. This mirrors
    exactly the runtime workaround (Pi extension prepends the same line),
    so training and inference see the same conditioning signal. Step-3.7's
    template only emits this line from a `reasoning_effort` render var that
    mlx_lm does not pass, so injecting into system content is the correct
    channel.
  * Split by source_trajectory_id (NOT by row): rows are cumulative-context
    slices of a trajectory, so a random row split would leak near-duplicate
    context across train/valid. Whole trajectories are held out.

Each row's messages already end on the target assistant turn -> train with
mask_prompt=True so loss is computed only on that final assistant response.

Usage:
    .venv/bin/python scripts/build_lora_data.py --out data/lora_step37 \
        --max-seq-length 8192 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

DEFAULT_MODEL = "/Users/true/.lmstudio/models/truemod/Step-3.7-p15-4bit-vblend-shared8"
FABLE_DIR = "data/lora_traces/fable5/data"
GPT56_JSONL = "data/lora_traces/gpt56sol/traces.jsonl"
KIMI_DIR = "data/lora_traces/kimi_k3/data"

LEVEL_MAP = {"max": "high", "xhigh": "high", "high": "high",
             "medium": "medium", "low": "low"}


def _truthy(v):
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    if hasattr(v, "__len__"):
        return len(v) > 0
    return bool(v)


def _to_plain(v):
    """parquet hands back numpy arrays / np scalars; make everything json-native."""
    if isinstance(v, np.ndarray):
        return [_to_plain(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_to_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_plain(x) for k, x in v.items()}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def norm_tool_call(tc):
    tc = _to_plain(dict(tc))
    fn = tc.get("function")
    if isinstance(fn, dict):
        a = fn.get("arguments")
        if isinstance(a, str):
            try:
                fn["arguments"] = json.loads(a) if a.strip() else {}
            except Exception:
                fn["arguments"] = {}
    else:
        a = tc.get("arguments")
        if isinstance(a, str):
            try:
                tc["arguments"] = json.loads(a) if a.strip() else {}
            except Exception:
                tc["arguments"] = {}
    return tc


def clean_messages(messages, level, tag_reasoning=True):
    """Normalize to template-safe messages. If tag_reasoning, inject the
    "Reasoning: <level>" line into the system message -- this is Step-3.7-
    specific syntax (matches its chat template's reasoning_effort render
    var), meaningless to a model that never saw that convention, so it must
    be OFF for any other target model (e.g. Gemma 4)."""
    out = []
    injected = False
    tag = f"Reasoning: {level}\n\n" if tag_reasoning else ""
    for m in messages:
        m = dict(m)
        role = m["role"]
        content = m.get("content", "") or ""
        if tag_reasoning and role == "system" and not injected:
            content = tag + content
            injected = True
        msg = {"role": role, "content": content}
        if _truthy(m.get("reasoning_content")):
            msg["reasoning_content"] = m["reasoning_content"]
        tcs = m.get("tool_calls")
        if _truthy(tcs):
            msg["tool_calls"] = [norm_tool_call(tc) for tc in tcs]
        if _truthy(m.get("tool_call_id")):
            msg["tool_call_id"] = m["tool_call_id"]
        if _truthy(m.get("name")):
            msg["name"] = m["name"]
        out.append(msg)
    if tag_reasoning and not injected:
        # no system message present -> prepend one carrying the tag
        out.insert(0, {"role": "system", "content": tag.rstrip()})
    return out


def norm_tools(tools_field):
    if not _truthy(tools_field):
        return None
    if isinstance(tools_field, str):
        try:
            return json.loads(tools_field)
        except Exception:
            return None
    return _to_plain(tools_field)


def load_fable():
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(f"{FABLE_DIR}/*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    return df.to_dict("records")


def load_gpt56():
    rows = []
    with open(GPT56_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_kimi():
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(f"{KIMI_DIR}/*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    return df.to_dict("records")


def build_records(rows, source, tag_reasoning=True):
    recs = []
    for r in rows:
        effort = r.get("reasoning_effort") or "high"
        level = LEVEL_MAP.get(str(effort), "high")
        msgs = clean_messages(list(r["messages"]), level, tag_reasoning=tag_reasoning)
        tools = norm_tools(r.get("tools"))
        traj = r.get("source_trajectory_id") or r.get("session_id") or f"{source}-{len(recs)}"
        recs.append({
            "messages": msgs,
            "tools": tools,
            "_traj": f"{source}:{traj}",
            "_level": level,
            "_category": r.get("category", ""),
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lora_step37")
    ap.add_argument("--max-seq-length", type=int, default=8192,
                    help="drop rows tokenizing beyond this (memory guard); "
                         "0 = keep all, report only")
    ap.add_argument("--valid-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="tokenizer/chat-template source. Use a local path "
                         "(tokenizer files only, weights not required) or an "
                         "HF repo id for a different target model.")
    ap.add_argument("--tag-reasoning", action="store_true", default=None,
                    help="inject 'Reasoning: <level>' into the system message. "
                         "Default: on for the Step-3.7 default model, off "
                         "otherwise (the tag is Step-3.7-specific syntax).")
    ap.add_argument("--no-tag-reasoning", dest="tag_reasoning", action="store_false")
    a = ap.parse_args()
    if a.tag_reasoning is None:
        a.tag_reasoning = (a.model == DEFAULT_MODEL)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    print(f"[data] tokenizer: {a.model} | tag_reasoning={a.tag_reasoning}", flush=True)

    print("[data] loading fable-5 ...", flush=True)
    recs = build_records(load_fable(), "fable5", a.tag_reasoning)
    print("[data] loading gpt-5.6-sol ...", flush=True)
    recs += build_records(load_gpt56(), "gpt56sol", a.tag_reasoning)
    print("[data] loading kimi-k3 ...", flush=True)
    recs += build_records(load_kimi(), "kimik3", a.tag_reasoning)
    print(f"[data] {len(recs)} raw records, "
          f"{len(set(r['_traj'] for r in recs))} trajectories", flush=True)
    print(f"[data] level mix: {dict(Counter(r['_level'] for r in recs))}", flush=True)

    # tokenize-length pass (also catches any late render failures).
    # NOTE: apply_chat_template(tokenize=True) returns a BatchEncoding whose
    # len() is the key count (2), NOT the token count -- so render to string
    # (tokenize=False) then encode without adding extra specials (the template
    # already emits them as text) to get the true token length.
    lengths = []
    kept = []
    render_fail = 0
    for i, r in enumerate(recs):
        try:
            rendered = tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=False)
            ids = tok(rendered, add_special_tokens=False)["input_ids"]
        except Exception as e:
            render_fail += 1
            if render_fail <= 5:
                print(f"  render fail: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        n = len(ids)
        lengths.append(n)
        if a.max_seq_length and n > a.max_seq_length:
            continue
        kept.append(r)
        if (i + 1) % 5000 == 0:
            print(f"  tokenized {i+1}/{len(recs)}", flush=True)

    lengths = np.array(lengths)
    print(f"[data] render failures: {render_fail}", flush=True)
    print(f"[data] token lengths: p50={np.percentile(lengths,50):.0f} "
          f"p90={np.percentile(lengths,90):.0f} p99={np.percentile(lengths,99):.0f} "
          f"max={lengths.max()}", flush=True)
    if a.max_seq_length:
        drop = (lengths > a.max_seq_length).sum()
        print(f"[data] dropped {drop} rows over max_seq_length={a.max_seq_length} "
              f"({100*drop/len(lengths):.1f}%); kept {len(kept)}", flush=True)

    # trajectory-level split
    rng = random.Random(a.seed)
    trajs = sorted(set(r["_traj"] for r in kept))
    rng.shuffle(trajs)
    n_val = int(len(trajs) * a.valid_frac)
    n_test = int(len(trajs) * a.test_frac)
    val_set = set(trajs[:n_val])
    test_set = set(trajs[n_val:n_val + n_test])
    splits = defaultdict(list)
    for r in kept:
        if r["_traj"] in val_set:
            splits["valid"].append(r)
        elif r["_traj"] in test_set:
            splits["test"].append(r)
        else:
            splits["train"].append(r)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                obj = {"messages": r["messages"]}
                if r["tools"] is not None:
                    obj["tools"] = r["tools"]
                f.write(json.dumps(obj) + "\n")
        print(f"[data] {name}: {len(rows)} rows, "
              f"levels={dict(Counter(r['_level'] for r in rows))}", flush=True)
    print(f"[data] wrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
