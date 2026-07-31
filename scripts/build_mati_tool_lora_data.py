"""Build train/valid/test.jsonl for mlx_lm.lora from mati agent traces.

Converts the dingo-wire-format turn logs (instruction/turns[].thought/
action/observation) from two sources into Step-3.7's native ChatDataset
shape ({"messages": [...], "tools": [...]}), same target schema as
build_lora_data.py:

  * mati_agent_31b_train.jsonl, curated_tier == "A_good_tool_workflow" only
    -- generic read/patch/verify workflows, not the SWE-bug-repair eval
    tiers (those are narrow-task, not general tool usage).
  * personal_gold_trajectories.jsonl -- real captured desktop usage +
    CTF trajectories. Trajectories containing any infra-only action type
    (submit_flag, retrieve_ctf, start_container, container_status,
    list_ctf_events, get_download_link, download_challenge_files,
    delegate_task(s)) are dropped whole, since those tools are specific to
    the CTF/orchestration harness and aren't general-purpose.

Each turn becomes an assistant message (thought -> content, action ->
tool_calls) followed by a tool-role message carrying the observation.
Turns with action.type == "none" are plain assistant answers (no
tool_calls). A single fixed tool-schema list (KEPT_TOOLS below) is
attached to every record.

Usage:
    .venv/bin/python scripts/build_mati_tool_lora_data.py \
        --out data/lora_mati_tools --max-seq-length 8192 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DEFAULT_MODEL = "/Users/true/.lmstudio/models/truemod/Step-3.7-p15-4bit-vblend-shared8"

MATI_31B_PATH = "/Users/true/Documents/GojoCode/data/curated/mati/mati_agent_31b_train.jsonl"
PERSONAL_GOLD_PATH = (
    "/Users/true/Documents/Mati_Train/curated/specialists/gemma26b/"
    "agentic/personal_gold_trajectories.jsonl"
)

INFRA_ONLY_TYPES = {
    "submit_flag", "retrieve_ctf", "start_container", "container_status",
    "list_ctf_events", "get_download_link", "download_challenge_files",
    "delegate_task", "delegate_tasks",
}

SYSTEM_PROMPT = (
    "You are a careful coding agent with access to a small set of "
    "workspace tools. Use them to read, write, and verify files, run "
    "commands, and search the codebase. Think before acting, and prefer "
    "the minimal set of calls needed to make progress."
)

KEPT_TOOLS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file, creating or overwriting it.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        }, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from the workspace.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "patch_file",
        "description": "Replace an exact substring in a file.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        }, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List files in a workspace directory.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"path": {"type": "string"}},
                        "required": []}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search files in the workspace for a regex pattern.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                        }, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the workspace.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "python",
        "description": "Run a Python script in the workspace sandbox.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "explore",
        "description": "Run a short Python snippet to inspect data or files.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "read_image",
        "description": "Load workspace image(s) for multimodal analysis.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "paths": {"type": "array", "items": {"type": "string"}},
                        }, "required": []}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch the contents of a URL.",
        "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"]}}},
]
KEPT_TOOL_NAMES = {t["function"]["name"] for t in KEPT_TOOLS}


def _norm_args(action_type: str, raw_input):
    """Coerce dingo's mixed input shapes (dict | str | None) into a dict
    matching the KEPT_TOOLS parameter schema for this action type."""
    if isinstance(raw_input, dict):
        return raw_input
    if raw_input in (None, ""):
        return {}
    if action_type in ("read_file", "list_dir"):
        return {"path": raw_input}
    if action_type == "bash":
        return {"command": raw_input}
    if action_type in ("python", "explore"):
        return {"code": raw_input}
    if action_type == "fetch_url":
        return {"url": raw_input}
    return {"value": raw_input}


def _obs_to_str(obs) -> str:
    if obs is None:
        return ""
    if isinstance(obs, str):
        return obs
    return json.dumps(obs)


def traj_to_messages(instruction: str, turns: list) -> list | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    for t in turns:
        action = t.get("action")
        thought = t.get("thought") or ""
        if not isinstance(action, dict):
            continue
        atype = action.get("type")
        if atype not in KEPT_TOOL_NAMES:
            if atype == "none":
                messages.append({"role": "assistant", "content": thought})
                continue
            return None  # unknown/dropped tool type -> discard whole trajectory
        call_id = f"toolu_{uuid.uuid4().hex[:24]}"
        messages.append({
            "role": "assistant",
            "content": thought,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": atype,
                    "arguments": _norm_args(atype, action.get("input")),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": atype,
            "content": _obs_to_str(t.get("observation")),
        })
    return messages


def load_mati_31b():
    recs = []
    with open(MATI_31B_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("curated_tier") != "A_good_tool_workflow":
                continue
            msgs = traj_to_messages(r["instruction"], r.get("turns", []))
            if msgs is None:
                continue
            recs.append({"messages": msgs, "tools": KEPT_TOOLS,
                          "_traj": f"mati31b:{r.get('task_id', len(recs))}"})
    return recs


def load_personal_gold():
    recs = []
    with open(PERSONAL_GOLD_PATH) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            turns = r.get("turns", [])
            types = {t.get("action", {}).get("type") for t in turns
                     if isinstance(t.get("action"), dict)}
            if types & INFRA_ONLY_TYPES:
                continue
            msgs = traj_to_messages(r["instruction"], turns)
            if msgs is None:
                continue
            recs.append({"messages": msgs, "tools": KEPT_TOOLS,
                          "_traj": f"persgold:{r.get('fingerprint', i)}"})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lora_mati_tools")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--valid-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    print(f"[data] tokenizer: {a.model}", flush=True)

    recs = load_mati_31b()
    print(f"[data] mati_31b A_good_tool_workflow: {len(recs)} trajectories", flush=True)
    pg = load_personal_gold()
    print(f"[data] personal_gold (general only): {len(pg)} trajectories", flush=True)
    recs += pg
    print(f"[data] {len(recs)} raw records, "
          f"{len(set(r['_traj'] for r in recs))} trajectories", flush=True)

    lengths = []
    kept = []
    render_fail = 0
    for r in recs:
        try:
            rendered = tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=False)
            ids = tok(rendered, add_special_tokens=False)["input_ids"]
        except Exception as e:
            render_fail += 1
            if render_fail <= 5:
                print(f"  render fail: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        n = len(ids)
        lengths.append(n)
        if a.max_seq_length and n > a.max_seq_length:
            continue
        kept.append(r)

    lengths = np.array(lengths) if lengths else np.array([0])
    print(f"[data] render failures: {render_fail}", flush=True)
    print(f"[data] token lengths: p50={np.percentile(lengths,50):.0f} "
          f"p90={np.percentile(lengths,90):.0f} p99={np.percentile(lengths,99):.0f} "
          f"max={lengths.max()}", flush=True)
    if a.max_seq_length:
        drop = (lengths > a.max_seq_length).sum()
        print(f"[data] dropped {drop} rows over max_seq_length={a.max_seq_length} "
              f"({100*drop/len(lengths):.1f}%); kept {len(kept)}", flush=True)

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
                f.write(json.dumps({"messages": r["messages"], "tools": r["tools"]}) + "\n")
        print(f"[data] {name}: {len(rows)} rows", flush=True)
    print(f"[data] wrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
