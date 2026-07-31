"""Validate that greghavens traces render through Step-3.7's chat template.

De-risks the LoRA data pipeline BEFORE building it: loads both datasets,
finds trajectories that exercise tool_calls + tool-response turns, and runs
them through the real Step-3.7 tokenizer's apply_chat_template exactly as
mlx_lm's ChatDataset will. If this passes, the data is training-ready with
only light normalization (parse `tools` JSON string, inject reasoning tag,
split by trajectory).
"""
from __future__ import annotations

import glob
import json
from collections import Counter

import pandas as pd
from transformers import AutoTokenizer

MODEL = "/Users/true/.lmstudio/models/truemod/Step-3.7-p15-4bit-vblend-shared8"
FABLE_DIR = "data/lora_traces/fable5/data"
GPT56_JSONL = "data/lora_traces/gpt56sol/traces.jsonl"


def load_fable():
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(f"{FABLE_DIR}/*.parquet"))]
    return pd.concat(frames, ignore_index=True).to_dict("records")


def load_gpt56():
    rows = []
    with open(GPT56_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_tools(tools_field):
    if tools_field is None:
        return None
    if isinstance(tools_field, str):
        tools_field = tools_field.strip()
        if not tools_field:
            return None
        return json.loads(tools_field)
    return tools_field


def _truthy(v):
    # parquet may hand back numpy arrays for list-typed cols; treat empty as falsy
    if v is None:
        return False
    if hasattr(v, "__len__"):
        return len(v) > 0
    return bool(v)


def clean_messages(messages):
    """Keep only keys the chat template consumes; drop empty tool_calls etc.
    so apply_chat_template's `tool_calls is defined` / mapping checks behave."""
    out = []
    for m in messages:
        m = dict(m)
        msg = {"role": m["role"], "content": m.get("content", "") or ""}
        if _truthy(m.get("reasoning_content")):
            msg["reasoning_content"] = m["reasoning_content"]
        if _truthy(m.get("tool_calls")):
            # numpy arrays of dicts -> plain list of dicts
            msg["tool_calls"] = [dict(tc) for tc in m["tool_calls"]]
        if _truthy(m.get("tool_call_id")):
            msg["tool_call_id"] = m["tool_call_id"]
        if _truthy(m.get("name")):
            msg["name"] = m["name"]
        out.append(msg)
    return out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    for name, loader in (("fable-5", load_fable), ("gpt-5.6-sol", load_gpt56)):
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        ds = loader()
        cols = sorted(ds[0].keys())
        print(f"rows: {len(ds)}  columns: {cols}")

        traj_key = "source_trajectory_id"
        if traj_key in ds[0]:
            trajs = Counter(r[traj_key] for r in ds)
            print(f"unique trajectories ({traj_key}): {len(trajs)}")
        if "reasoning_effort" in ds[0]:
            print(f"reasoning_effort values: {dict(Counter(r['reasoning_effort'] for r in ds))}")
        if "category" in ds[0]:
            cats = Counter(r["category"] for r in ds)
            print(f"categories ({len(cats)}): {dict(cats.most_common(8))} ...")

        # find rows that exercise the hard parts: tool_calls and tool-role turns
        n_test = 0
        n_with_toolcalls = 0
        n_with_toolrole = 0
        n_render_fail = 0
        last_roles = Counter()
        for i in range(len(ds)):
            row = ds[i]
            msgs = clean_messages(row["messages"])
            tools = norm_tools(row.get("tools"))
            last_roles[msgs[-1]["role"]] += 1
            has_tc = any(m.get("tool_calls") for m in msgs)
            has_tr = any(m["role"] == "tool" for m in msgs)
            n_with_toolcalls += has_tc
            n_with_toolrole += has_tr
            # render a spread: first 50, plus every tool-heavy one up to 400 tests
            if i < 50 or (has_tc or has_tr):
                if n_test >= 400:
                    continue
                n_test += 1
                try:
                    tok.apply_chat_template(msgs, tools=tools, tokenize=True)
                except Exception as e:
                    n_render_fail += 1
                    if n_render_fail <= 3:
                        print(f"  RENDER FAIL row {i}: {type(e).__name__}: {str(e)[:160]}")

        print(f"rows with tool_calls: {n_with_toolcalls}/{len(ds)}  "
              f"tool-role turns: {n_with_toolrole}/{len(ds)}")
        print(f"last-message roles: {dict(last_roles)}")
        print(f"apply_chat_template: tested {n_test}, failed {n_render_fail}")


if __name__ == "__main__":
    main()
