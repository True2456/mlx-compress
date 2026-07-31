"""Reconstruct full, unsliced agentic trajectories for latent-looping
curriculum training.

build_lora_data.py deliberately slices each trajectory into one row per
assistant turn (derivation: cumulative-next-assistant-v1) -- correct for
ordinary SFT, where you want many independent (context, target-turn) pairs.
That slicing throws away the very multi-step reasoning-and-acting structure
COCONUT's curriculum needs: a single trajectory with several assistant turns
to progressively compress into continuous thought vectors.

This reverses that: for each source_trajectory_id, keep only the fullest
available slice (max n_messages), which by construction of
cumulative-next-assistant-v1 is the row ending on the final assistant turn --
i.e. the complete original trajectory in one row. Filters to trajectories
with enough assistant turns to actually have something to compress.

Usage:
    .venv/bin/python scripts/build_loop_curriculum_data.py \
        --out data/loop_curriculum_data --min-assistant-turns 3
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

FABLE_DIR = "data/lora_traces/fable5/data"
GPT56_JSONL = "data/lora_traces/gpt56sol/traces.jsonl"
KIMI_DIR = "data/lora_traces/kimi_k3/data"


def _truthy(v):
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    if hasattr(v, "__len__"):
        return len(v) > 0
    return bool(v)


def norm_tool_call(tc):
    """Same fix as build_lora_data.py's norm_tool_call: arguments arrive as
    a JSON-encoded string, not a dict -- apply_chat_template chokes on that."""
    tc = _to_plain(dict(tc))
    fn = tc.get("function")
    if isinstance(fn, dict):
        arg = fn.get("arguments")
        if isinstance(arg, str):
            try:
                fn["arguments"] = json.loads(arg) if arg.strip() else {}
            except Exception:
                fn["arguments"] = {}
    else:
        arg = tc.get("arguments")
        if isinstance(arg, str):
            try:
                tc["arguments"] = json.loads(arg) if arg.strip() else {}
            except Exception:
                tc["arguments"] = {}
    return tc


def norm_tools(tools_field):
    """tools itself also arrives as a JSON-encoded string, same issue."""
    if not _truthy(tools_field):
        return None
    if isinstance(tools_field, str):
        try:
            return json.loads(tools_field)
        except Exception:
            return None
    return _to_plain(tools_field)


def clean_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        tcs = m.get("tool_calls")
        if _truthy(tcs):
            m["tool_calls"] = [norm_tool_call(tc) for tc in tcs]
        out.append(m)
    return out


def _to_plain(v):
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


def fullest_slice_per_trajectory(rows: list[dict], source: str) -> list[dict]:
    by_traj: dict[str, dict] = {}
    for r in rows:
        if r.get("derivation") != "cumulative-next-assistant-v1":
            continue
        traj = r.get("source_trajectory_id")
        if traj is None:
            continue
        n = r.get("n_messages", 0)
        cur = by_traj.get(traj)
        if cur is None or n > cur.get("n_messages", 0):
            by_traj[traj] = r

    out = []
    mismatched = 0
    for traj, r in by_traj.items():
        n = r.get("n_messages", 0)
        orig = r.get("original_n_messages", n)
        if n != orig:
            mismatched += 1
            continue
        out.append(r)
    if mismatched:
        print(f"[{source}] dropped {mismatched} trajectories where the fullest "
              f"available slice didn't reach original_n_messages", flush=True)
    return out


def count_assistant_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/loop_curriculum_data")
    ap.add_argument("--min-assistant-turns", type=int, default=3,
                     help="drop trajectories with fewer assistant turns than this "
                          "-- need genuine multi-step structure to compress")
    ap.add_argument("--valid-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("[data] loading fable-5 ...", flush=True)
    fable_rows = fullest_slice_per_trajectory(load_fable(), "fable5")
    print("[data] loading gpt-5.6-sol ...", flush=True)
    gpt56_rows = fullest_slice_per_trajectory(load_gpt56(), "gpt56sol")
    print("[data] loading kimi-k3 ...", flush=True)
    kimi_rows = fullest_slice_per_trajectory(load_kimi(), "kimi_k3")

    all_trajs = []
    for source, rows in [("fable5", fable_rows), ("gpt56sol", gpt56_rows), ("kimi_k3", kimi_rows)]:
        kept = 0
        for r in rows:
            messages = clean_messages(_to_plain(r["messages"]))
            n_asst = count_assistant_turns(messages)
            if n_asst < a.min_assistant_turns:
                continue
            all_trajs.append({
                "messages": messages,
                "tools": norm_tools(r.get("tools")),
                "_traj": f"{source}:{r['source_trajectory_id']}",
                "_source": source,
                "_n_assistant_turns": n_asst,
            })
            kept += 1
        print(f"[data] {source}: {len(rows)} full trajectories, {kept} with "
              f">={a.min_assistant_turns} assistant turns", flush=True)

    print(f"[data] total usable trajectories: {len(all_trajs)}", flush=True)
    turn_counts = [t["_n_assistant_turns"] for t in all_trajs]
    if turn_counts:
        print(f"[data] assistant-turn counts: p50={np.percentile(turn_counts,50):.0f} "
              f"p90={np.percentile(turn_counts,90):.0f} max={max(turn_counts)}", flush=True)

    rng = random.Random(a.seed)
    rng.shuffle(all_trajs)
    n_val = int(len(all_trajs) * a.valid_frac)
    n_test = int(len(all_trajs) * a.test_frac)
    splits = {
        "valid": all_trajs[:n_val],
        "test": all_trajs[n_val:n_val + n_test],
        "train": all_trajs[n_val + n_test:],
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                obj = {"messages": r["messages"]}
                if r["tools"] is not None:
                    obj["tools"] = r["tools"]
                f.write(json.dumps(obj) + "\n")
        print(f"[data] {name}: {len(rows)} trajectories", flush=True)
    print(f"[data] wrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
