"""Build train/valid/test.jsonl for mlx_lm.lora from local Claude Code session
transcripts (~/.claude/projects/*/*.jsonl).

Same target schema as build_lora_data.py: mlx_lm's native ChatDataset format
({"messages": [...], "tools": [...]}), tokenize-length filtered and split by
trajectory (session), never by row.

Claude Code transcripts are a parentUuid-linked tree (branches appear from
edits/retries/compaction). For each session we walk from the most recent
leaf back to the root along isSidechain=False nodes only -- this selects the
single kept main thread and silently drops abandoned branches, rather than
training on dead-end attempts.

Content-block mapping:
  * assistant "thinking" block  -> message["reasoning_content"]
  * assistant "text" blocks     -> message["content"] (joined)
  * assistant "tool_use" blocks -> message["tool_calls"] (OpenAI-style)
  * user "tool_result" blocks   -> one role:"tool" message per block,
    matched back to its tool name via the tool_use id seen earlier

There is no logged system prompt or tool JSON-schema in these transcripts
(the harness injects both at runtime and doesn't persist them to disk), so
rows are written with tools=None -- same as any non-tool-schema SFT row in
build_lora_data.py.

Usage:
    .venv/bin/python scripts/build_claude_code_lora_data.py \
        --out data/lora_claude_code --max-seq-length 8192 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DEFAULT_MODEL = "/Users/true/.lmstudio/models/truemod/Step-3.7-p15-4bit-vblend-shared8"
DEFAULT_LOGS_GLOB = str(Path.home() / ".claude/projects/*/*.jsonl")
MAX_TOOL_RESULT_CHARS = 2000
MAX_TEXT_CHARS = 4000
MAX_CONTEXT_MESSAGES = 24
META_TYPES = {"queue-operation", "custom-title", "ai-title", "last-prompt",
              "attachment", "mode", "system", "summary"}


def _truncate(s, n=MAX_TOOL_RESULT_CHARS):
    if len(s) <= n:
        return s
    return s[:n] + f"\n...[truncated {len(s)-n} chars]"


def _stringify_tool_result(content):
    if isinstance(content, str):
        return _truncate(content)
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
            elif isinstance(blk, dict) and blk.get("type") == "image":
                parts.append("[image omitted]")
        return _truncate("\n".join(parts))
    return _truncate(json.dumps(content))


def load_session(path):
    """Return (all_nodes, order) for one session, isSidechain records
    dropped but meta record types (stop_hook_summary etc.) KEPT -- they can
    sit inside the parentUuid chain between two message records, so
    dropping them here would fragment the walk in main_thread()."""
    nodes = {}
    order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("isSidechain"):
                continue
            uuid = o.get("uuid")
            if not uuid:
                continue
            nodes[uuid] = o
            order.append(uuid)
    return nodes, order


def main_thread(nodes, order):
    """Walk parentUuid links from the last record in file order back to
    root, over the full (message + meta) graph, then keep only user/
    assistant records for the returned chain."""
    if not order:
        return []
    leaf = order[-1]
    chain = []
    u = leaf
    seen = set()
    while u is not None and u in nodes and u not in seen:
        seen.add(u)
        chain.append(u)
        u = nodes[u].get("parentUuid")
    chain.reverse()
    return [nodes[u] for u in chain if nodes[u].get("type") in ("user", "assistant")]


def convert_thread(records):
    """Convert a Claude Code record chain to a flat OpenAI-style message
    list, returning (messages, tool_id_to_name)."""
    messages = []
    tool_names = {}
    for rec in records:
        msg = rec["message"]
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            if isinstance(content, str):
                if content.strip():
                    messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "tool_result":
                        tcid = blk.get("tool_use_id", "")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tcid,
                            "name": tool_names.get(tcid, ""),
                            "content": _stringify_tool_result(blk.get("content", "")),
                        })
                    elif blk.get("type") == "text" and blk.get("text", "").strip():
                        messages.append({"role": "user", "content": blk["text"]})
        elif role == "assistant":
            if isinstance(content, str):
                if content.strip():
                    messages.append({"role": "assistant", "content": content})
                continue
            if not isinstance(content, list):
                continue
            text_parts = []
            reasoning_parts = []
            tool_calls = []
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                bt = blk.get("type")
                if bt == "text":
                    text_parts.append(blk.get("text", ""))
                elif bt == "thinking":
                    reasoning_parts.append(blk.get("thinking", ""))
                elif bt == "tool_use":
                    tcid = blk.get("id", "")
                    name = blk.get("name", "")
                    tool_names[tcid] = name
                    tool_calls.append({
                        "id": tcid,
                        "type": "function",
                        "function": {"name": name, "arguments": blk.get("input", {})},
                    })
            if not text_parts and not tool_calls:
                continue
            out = {"role": "assistant", "content": _truncate("\n".join(text_parts), MAX_TEXT_CHARS)}
            if reasoning_parts:
                out["reasoning_content"] = _truncate("\n".join(reasoning_parts), MAX_TEXT_CHARS)
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)
    return messages


def _window(messages, end):
    """Trailing slice messages[:end] capped at MAX_CONTEXT_MESSAGES, backed
    up so it never starts on a "tool" message (which would be orphaned
    without the assistant tool_call that preceded it)."""
    start = max(0, end - MAX_CONTEXT_MESSAGES)
    while start > 0 and messages[start]["role"] == "tool":
        start -= 1
    return messages[start:end]


def build_records_for_session(path):
    """Yield one record (trailing-window context row ending on an assistant
    turn) per assistant turn in the session's main thread. A window, not
    the full from-session-start prefix, because real sessions run hundreds
    of turns -- unwindowed cumulative context blows past any max_seq_length
    within ~10 turns and almost every row gets dropped."""
    nodes, order = load_session(path)
    thread = main_thread(nodes, order)
    if not thread:
        return []
    messages = convert_thread(thread)
    traj = Path(path).stem
    recs = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        recs.append({
            "messages": _window(messages, i + 1),
            "tools": None,
            "_traj": f"claude_code:{traj}",
            "_level": "high",
            "_category": "claude_code_session",
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lora_claude_code")
    ap.add_argument("--logs-glob", default=DEFAULT_LOGS_GLOB)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--valid-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    print(f"[data] tokenizer: {a.model}", flush=True)

    paths = sorted(glob.glob(a.logs_glob))
    print(f"[data] {len(paths)} session files matched {a.logs_glob}", flush=True)

    recs = []
    for p in paths:
        try:
            recs += build_records_for_session(p)
        except Exception as e:
            print(f"  skip {p}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"[data] {len(recs)} raw records, "
          f"{len(set(r['_traj'] for r in recs))} sessions", flush=True)

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

    lengths = np.array(lengths) if lengths else np.array([0])
    print(f"[data] render failures: {render_fail}", flush=True)
    print(f"[data] token lengths: p50={np.percentile(lengths,50):.0f} "
          f"p90={np.percentile(lengths,90):.0f} p99={np.percentile(lengths,99):.0f} "
          f"max={lengths.max()}", flush=True)

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
        print(f"[data] {name}: {len(rows)} rows", flush=True)
    print(f"[data] wrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
