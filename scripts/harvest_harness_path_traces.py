#!/usr/bin/env python3
"""Build Step-3.7 harness path-fidelity training data.

Mix (defaults):
  ~85% first-try-correct synthetic successes (Codex / Pi / Claude adapters)
  ~10% short one-step repair after a real tool error
  ~5% longer clean multi-step success (edit → run → view)

Also converts live Codex rollouts / Pi sessions / Claude Code transcripts
when present, then hard-filters listdir loops and same-path retries.

Outputs ChatDataset rows: {"messages": [...], "tools": [...], ...meta}
plus optional ORPO pairs where live bad turns become rejected.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "lora_step37_harness"
HARVEST_CWD = "/Users/true/Downloads/trace-harvest"

CODEX_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "View an image file at an absolute or workspace-relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Exact filesystem path to the image."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "workdir": {"type": "string"},
                    "yield_time_ms": {"type": "integer"},
                    "max_output_tokens": {"type": "integer"},
                },
                "required": ["cmd"],
            },
        },
    },
]

PI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text or image file. For images, pass the exact path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]

CLAUDE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file from the local filesystem. Images are supported.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a bash command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]

SYSTEM = {
    "codex": (
        "You are an autonomous coding agent working in a real repository on the user's machine. "
        "Use view_image for images and exec_command for shell. Prefer exact paths. "
        "On a missing-file error, correct the path once; do not loop directory listings."
    ),
    "pi": (
        "You are an autonomous coding agent running on the user's machine. "
        "Use read_file for files/images and bash for shell. Prefer exact paths. "
        "On a missing-file error, correct the path once; do not loop directory listings."
    ),
    "claude": (
        "You are Claude Code, an autonomous coding agent. "
        "Use Read for files/images and Bash for shell. Prefer exact paths. "
        "On a missing-file error, correct the path once; do not loop directory listings."
    ),
}

# Diverse path-fidelity families (forest renders are only one slice).
# Each entry: family, rel path, modality (image|text|shell), user ask, optional bad_rel.
TASK_SPECS = [
    # images — forest (kept small)
    {"family": "forest-render", "rel": "renders/iter-03.png", "modality": "image",
     "ask": "open this exact image and say if it looks like a forest scene"},
    {"family": "forest-render", "rel": "renders/iter-05.png", "modality": "image",
     "ask": "open this exact image and say if it looks like a forest scene"},
    # images — other naming
    {"family": "asset-frame", "rel": "assets/shots/frame-01.png", "modality": "image",
     "ask": "open this exact screenshot and say what you see in one sentence"},
    {"family": "asset-frame", "rel": "assets/shots/frame-02.png", "modality": "image",
     "ask": "open this exact screenshot and say what you see in one sentence"},
    # configs / data / logs / code
    {"family": "config", "rel": "configs/server.toml", "modality": "text",
     "ask": "read this config and report the port value"},
    {"family": "batch-meta", "rel": "data/batch_07/meta.json", "modality": "text",
     "ask": "read this JSON and report the batch_id"},
    {"family": "log", "rel": "logs/run-03.log", "modality": "text",
     "ask": "read this log and report the status field"},
    {"family": "report", "rel": "logs/report-03.md", "modality": "text",
     "ask": "read this report and summarize it in one short sentence"},
    {"family": "source", "rel": "src/math_util.py", "modality": "text",
     "ask": "read this file and say what function it defines"},
    {"family": "test", "rel": "tests/test_math_util.py", "modality": "text",
     "ask": "read this test file and say what it asserts"},
    # shell with exact paths
    {"family": "pytest", "rel": "tests/test_math_util.py", "modality": "shell",
     "ask": "run pytest on this exact test file path and report pass/fail",
     "shell_cmd": "python3 -m pytest tests/test_math_util.py -q"},
    {"family": "py-compile", "rel": "src/math_util.py", "modality": "shell",
     "ask": "compile-check this exact source path and report ok/error",
     "shell_cmd": "python3 -m py_compile src/math_util.py && echo OK"},
]

# Underscore / padding hallucinations paired with correct relatives.
REPAIR_SPECS = [
    {"family": "forest-render", "bad": "renders/iter-0_3.png", "good": "renders/iter-03.png",
     "modality": "image", "hint": "iteration 3 render"},
    {"family": "asset-frame", "bad": "assets/shots/frame-0_1.png", "good": "assets/shots/frame-01.png",
     "modality": "image", "hint": "frame 1 screenshot"},
    {"family": "batch-meta", "bad": "data/batch_7/meta.json", "good": "data/batch_07/meta.json",
     "modality": "text", "hint": "batch 7 metadata"},
    {"family": "log", "bad": "logs/run-3.log", "good": "logs/run-03.log",
     "modality": "text", "hint": "run 3 log"},
    {"family": "report", "bad": "logs/report-3.md", "good": "logs/report-03.md",
     "modality": "text", "hint": "report 03"},
    {"family": "config", "bad": "config/server.toml", "good": "configs/server.toml",
     "modality": "text", "hint": "server config"},
]

LONG_SPECS = [
    {"family": "latest-render", "pointer": "renders/LATEST.txt", "target": "renders/iter-05.png",
     "modality": "image", "ask": "read the pointer file, open that exact image, one-sentence summary"},
    {"family": "latest-asset", "pointer": "assets/LATEST.txt", "target": "assets/shots/frame-02.png",
     "modality": "image", "ask": "read the pointer file, open that exact image, one-sentence summary"},
    {"family": "edit-test", "pointer": "src/math_util.py", "target": "tests/test_math_util.py",
     "modality": "text", "ask": "read the source, then the matching test, and say if the test matches the function"},
    {"family": "config-run", "pointer": "configs/server.toml", "target": "logs/run-03.log",
     "modality": "text", "ask": "read the server config, then the run log, and say if status is ok"},
]


def _tc(name: str, arguments: dict, call_id: str | None = None) -> dict:
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _row(harness: str, messages: list, tools: list, kind: str, task: str, family: str = "") -> dict:
    return {
        "messages": messages,
        "tools": tools,
        "harness": harness,
        "kind": kind,
        "task": task,
        "family": family,
        "domain": "coding",
        "teacher_model": "synthetic-gold-v1",
        "trace_format": f"{harness}-adapter",
    }


def _harness_kit(harness: str):
    if harness == "codex":
        return CODEX_TOOLS, SYSTEM["codex"]
    if harness == "pi":
        return PI_TOOLS, SYSTEM["pi"]
    return CLAUDE_TOOLS, SYSTEM["claude"]


def _path_tool(harness: str, abs_path: str, modality: str, call_id: str | None = None):
    """Return (tool_name, args) for reading/viewing a path in this harness."""
    if harness == "codex":
        if modality == "image":
            return "view_image", {"path": abs_path}, _tc("view_image", {"path": abs_path}, call_id)
        if modality == "shell":
            # shell uses exec separately
            return "exec_command", {"cmd": abs_path}, None
        return "exec_command", {"cmd": f"cat {abs_path}", "workdir": HARVEST_CWD, "max_output_tokens": 800}, _tc(
            "exec_command",
            {"cmd": f"cat {abs_path}", "workdir": HARVEST_CWD, "max_output_tokens": 800},
            call_id,
        )
    if harness == "pi":
        if modality == "shell":
            return "bash", {"command": abs_path}, None
        return "read_file", {"path": abs_path}, _tc("read_file", {"path": abs_path}, call_id)
    if modality == "shell":
        return "Bash", {"command": abs_path}, None
    return "Read", {"file_path": abs_path}, _tc("Read", {"file_path": abs_path}, call_id)


def _shell_tool(harness: str, cmd: str, call_id: str | None = None):
    if harness == "codex":
        return _tc(
            "exec_command",
            {"cmd": cmd, "workdir": HARVEST_CWD, "max_output_tokens": 1200},
            call_id,
        )
    if harness == "pi":
        return _tc("bash", {"command": f"cd {HARVEST_CWD} && {cmd}"}, call_id)
    return _tc("Bash", {"command": f"cd {HARVEST_CWD} && {cmd}"}, call_id)


def synth_first_try(harness: str, spec: dict) -> dict:
    tools, sys = _harness_kit(harness)
    rel = spec["rel"]
    abs_path = f"{HARVEST_CWD}/{rel}"
    modality = spec["modality"]
    user = (
        f"In {HARVEST_CWD}, {spec['ask']}. Exact path: `{rel}` "
        f"(absolute ok: `{abs_path}`). Do not invent alternate spellings or underscores."
    )
    if modality == "shell":
        asst = {
            "role": "assistant",
            "content": f"Running the check on exact path `{rel}`.",
            "tool_calls": [_shell_tool(harness, spec["shell_cmd"])],
        }
    else:
        _, _, tc = _path_tool(harness, abs_path, modality)
        asst = {
            "role": "assistant",
            "content": f"Using the exact path `{rel}`.",
            "tool_calls": [tc],
        }
    return _row(
        harness,
        [{"role": "system", "content": sys}, {"role": "user", "content": user}, asst],
        tools,
        "first_try_correct",
        f"{spec['family']}:{Path(rel).name}",
        family=spec["family"],
    )


def synth_repair(harness: str, spec: dict) -> dict:
    tools, sys = _harness_kit(harness)
    bad_abs = f"{HARVEST_CWD}/{spec['bad']}"
    good_abs = f"{HARVEST_CWD}/{spec['good']}"
    modality = spec["modality"]
    err = f"Error: No such file or directory: {bad_abs}"
    id1, id2 = f"call_{uuid.uuid4().hex[:12]}", f"call_{uuid.uuid4().hex[:12]}"
    _, _, tc_bad = _path_tool(harness, bad_abs, modality, id1)
    _, _, tc_good = _path_tool(harness, good_abs, modality, id2)
    user = (
        f"In {HARVEST_CWD}, open the {spec['hint']}. Someone suggested `{spec['bad']}`; "
        f"use the real path if that fails. Correct once — no listing loops."
    )
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": f"Trying `{spec['bad']}` first.",
            "tool_calls": [tc_bad],
        },
        {"role": "tool", "tool_call_id": id1, "content": err},
        {
            "role": "assistant",
            "content": f"Path was wrong. Correcting once to `{spec['good']}`.",
            "tool_calls": [tc_good],
        },
    ]
    return _row(
        harness,
        messages,
        tools,
        "one_step_repair",
        f"repair:{spec['family']}",
        family=spec["family"],
    )


def synth_long_success(harness: str, spec: dict) -> dict:
    tools, sys = _harness_kit(harness)
    ptr = f"{HARVEST_CWD}/{spec['pointer']}"
    tgt = f"{HARVEST_CWD}/{spec['target']}"
    modality = spec["modality"]
    id1, id2 = f"call_{uuid.uuid4().hex[:12]}", f"call_{uuid.uuid4().hex[:12]}"
    _, _, tc1 = _path_tool(harness, ptr, "text", id1)
    _, _, tc2 = _path_tool(harness, tgt, modality, id2)
    tool_out = spec["target"] + "\n" if "LATEST" in spec["pointer"] else f"[contents of {spec['pointer']}]\n"
    messages = [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": (
                f"In {HARVEST_CWD}, {spec['ask']}. Exact paths only; no listing loops."
            ),
        },
        {
            "role": "assistant",
            "content": f"Reading `{spec['pointer']}` first.",
            "tool_calls": [tc1],
        },
        {"role": "tool", "tool_call_id": id1, "content": tool_out},
        {
            "role": "assistant",
            "content": f"Next: exact path `{spec['target']}`.",
            "tool_calls": [tc2],
        },
    ]
    return _row(
        harness,
        messages,
        tools,
        "long_success",
        f"long:{spec['family']}",
        family=spec["family"],
    )


def build_synthetic(n_first: int, n_repair: int, n_long: int, seed: int) -> list:
    rng = random.Random(seed)
    harnesses = ["codex", "pi", "claude"]
    rows = []
    for i in range(n_first):
        h = harnesses[i % 3]
        spec = TASK_SPECS[i % len(TASK_SPECS)]
        rows.append(synth_first_try(h, spec))
    for i in range(n_repair):
        h = harnesses[i % 3]
        rows.append(synth_repair(h, REPAIR_SPECS[i % len(REPAIR_SPECS)]))
    for i in range(n_long):
        h = harnesses[i % 3]
        rows.append(synth_long_success(h, LONG_SPECS[i % len(LONG_SPECS)]))
    rng.shuffle(rows)
    return rows


# ---- live session converters -------------------------------------------------

LISTDIR_RE = re.compile(r"os\.listdir|\bls\b.*renders|listdir\(", re.I)
BAD_PATH_RE = re.compile(r"iter-0_\d+")


def _args_obj(arguments):
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except Exception:
            return {"_raw": arguments}
    return {}


def is_listdir_loop(messages: list) -> bool:
    cmds = []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = _args_obj(fn.get("arguments"))
            blob = json.dumps(args)
            if LISTDIR_RE.search(blob):
                cmds.append(blob)
    return len(cmds) >= 3


def has_bad_path_hallucination(messages: list) -> bool:
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if BAD_PATH_RE.search(json.dumps(fn.get("arguments"))):
                # allow if later corrected in same trajectory and kind is repair
                return True
    return False


def first_try_correct_path(messages: list) -> bool:
    """True if first image/path tool call uses a real zero-padded iter-NN path."""
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args = _args_obj(fn.get("arguments"))
            blob = json.dumps(args)
            if name in ("view_image", "read_file", "Read") or "path" in args or "file_path" in args:
                if BAD_PATH_RE.search(blob):
                    return False
                if re.search(r"iter-\d{2}\.png|reference\.jpeg", blob):
                    return True
    return False


def flatten_codex_rollout(path: str) -> list:
    """Flatten Codex rollout events into OpenAI-style messages (best effort)."""
    messages = [{"role": "system", "content": SYSTEM["codex"]}]
    pending_user = None
    for line in open(path):
        try:
            o = json.loads(line)
        except Exception:
            continue
        p = o.get("payload") or {}
        t = p.get("type") or o.get("type")
        if t == "message" and p.get("role") == "user":
            texts = [
                c.get("text", "")
                for c in p.get("content", [])
                if isinstance(c, dict) and c.get("type") in ("input_text", "text")
            ]
            text = "\n".join(t for t in texts if t).strip()
            if text:
                pending_user = text
        if t == "function_call":
            if pending_user:
                messages.append({"role": "user", "content": pending_user})
                pending_user = None
            args = p.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(p.get("name", "unknown"), args or {}, p.get("call_id") or p.get("id"))
                    ],
                }
            )
        if t == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": p.get("call_id") or "",
                    "content": str(p.get("output") or p.get("content") or "")[:2000],
                }
            )
    return messages


def convert_codex_rollout(path: str) -> list[dict]:
    messages = flatten_codex_rollout(path)
    if len(messages) < 3:
        return []
    base = {
        "tools": CODEX_TOOLS,
        "harness": "codex",
        "domain": "coding",
        "teacher_model": "step-3.7-live",
        "trace_format": "codex-rollout",
        "source_path": path,
    }
    return _slice_cumulative(messages, base, Path(path).stem)


def _final_tool_blob(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            return json.dumps((tc.get("function") or {}).get("arguments"))
    return ""


def _final_tool_is_good_path(messages: list) -> bool:
    blob = _final_tool_blob(messages)
    if not blob:
        return False
    if BAD_PATH_RE.search(blob):
        return False
    return bool(re.search(r"iter-\d{2}\.png|reference\.jpeg|LATEST\.txt", blob))


def _slice_cumulative(messages: list, base: dict, task: str) -> list[dict]:
    """Emit cumulative-next-assistant rows ending on each tool-calling assistant turn."""
    out = []
    asst_idxs = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    for step, idx in enumerate(asst_idxs, 1):
        prefix = messages[: idx + 1]
        if is_listdir_loop(prefix):
            continue
        # Never train on a turn whose *target* tool call still has a bad path.
        if BAD_PATH_RE.search(_final_tool_blob(prefix)):
            continue
        kind = "live_rollout"
        if first_try_correct_path(prefix) and not has_bad_path_hallucination(prefix):
            kind = "live_first_try"
        elif _repair_recovers(prefix) and _final_tool_is_good_path(prefix):
            kind = "live_repair"
        elif _final_tool_is_good_path(prefix):
            kind = "live_first_try"
        row = {
            **base,
            "messages": prefix,
            "kind": kind,
            "task": f"{task}:step{step}",
            "derivation": "cumulative-next-assistant-v1",
            "assistant_step": step,
            "assistant_steps": len(asst_idxs),
            "n_messages": len(prefix),
        }
        out.append(row)
    if not out:
        out.append(
            {
                **base,
                "messages": messages,
                "kind": "live_rollout",
                "task": task,
            }
        )
    return out


def convert_claude_session(path: str) -> list[dict]:
    """Best-effort Claude Code transcript → OpenAI messages."""
    nodes = {}
    order = []
    for line in open(path):
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
    if not order:
        return []
    # walk back from last message
    chain = []
    cur = order[-1]
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        n = nodes.get(cur)
        if not n:
            break
        chain.append(n)
        cur = n.get("parentUuid")
    chain.reverse()
    messages = [{"role": "system", "content": SYSTEM["claude"]}]
    for n in chain:
        typ = n.get("type")
        msg = n.get("message") or {}
        role = msg.get("role") or typ
        content = msg.get("content")
        if role == "user":
            if isinstance(content, list):
                texts = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        texts.append(b.get("text", ""))
                    elif b.get("type") == "tool_result":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": b.get("tool_use_id", ""),
                                "content": _stringify_tool_result(b.get("content"))[:2000],
                            }
                        )
                text = "\n".join(t for t in texts if t).strip()
                if text:
                    messages.append({"role": "user", "content": text})
            elif isinstance(content, str) and content.strip():
                messages.append({"role": "user", "content": content.strip()})
        elif role == "assistant":
            text_parts, tcs = [], []
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif b.get("type") == "tool_use":
                        tcs.append(
                            _tc(b.get("name", ""), b.get("input") or {}, b.get("id"))
                        )
            out = {"role": "assistant", "content": "\n".join(text_parts).strip()}
            if tcs:
                out["tool_calls"] = tcs
            if out["content"] or tcs:
                messages.append(out)
    if len(messages) < 3:
        return []
    base = {
        "tools": CLAUDE_TOOLS,
        "harness": "claude",
        "domain": "coding",
        "teacher_model": "step-3.7-live",
        "trace_format": "claude-code",
        "source_path": path,
    }
    return _slice_cumulative(messages, base, Path(path).stem)


def _stringify_tool_result(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, dict) and b.get("type") == "image":
                parts.append("[image omitted]")
        return "\n".join(parts)
    return json.dumps(content)


def convert_pi_session(path: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM["pi"]}]
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "message":
            continue
        msg = d.get("message") or {}
        role = msg.get("role")
        blocks = msg.get("content") or []
        if not isinstance(blocks, list):
            continue
        if role == "user":
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            if text:
                msgs.append({"role": "user", "content": text})
        elif role == "assistant":
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            tcs = []
            for b in blocks:
                if b.get("type") == "toolCall":
                    tcs.append(
                        _tc(b.get("name", ""), b.get("arguments") or {}, b.get("id"))
                    )
            out = {"role": "assistant", "content": text}
            if tcs:
                out["tool_calls"] = tcs
            msgs.append(out)
        elif role == "toolResult":
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            msgs.append({"role": "tool", "content": text[:2000]})
    if len(msgs) < 3:
        return []
    base = {
        "tools": PI_TOOLS,
        "harness": "pi",
        "domain": "coding",
        "teacher_model": "step-3.7-live",
        "trace_format": "pi-session",
        "source_path": path,
    }
    return _slice_cumulative(msgs, base, Path(path).stem)


def filter_keep(row: dict) -> tuple[bool, str]:
    kind = row.get("kind")
    msgs = row.get("messages") or []
    if kind in ("first_try_correct", "long_success"):
        return True, "synthetic"
    if kind == "one_step_repair":
        return True, "synthetic_repair"
    blob_all = json.dumps(msgs)
    if "renders/game/renders" in blob_all:
        return False, "confused_nested_path"
    if is_listdir_loop(msgs):
        return False, "listdir_loop"
    if kind in ("live_first_try", "live_repair"):
        names = [
            ((tc.get("function") or {}).get("name") or "")
            for m in msgs
            for tc in (m.get("tool_calls") or [])
        ]
        final_name = names[-1] if names else ""
        final_blob = _final_tool_blob(msgs)
        if final_name in {"view_image", "read_file", "Read", "read"} and _final_tool_is_good_path(msgs):
            return True, kind
        if final_name in ("exec_command", "bash", "Bash") and "LATEST.txt" in final_blob:
            return True, "live_latest_read"
        return False, "live_non_path_tool"
    if kind.startswith("live") and has_bad_path_hallucination(msgs) and not _repair_recovers(msgs):
        return False, "bad_path_unrecovered"
    if kind.startswith("live") and first_try_correct_path(msgs):
        return True, "live_first_try"
    if kind.startswith("live") and _repair_recovers(msgs):
        return True, "live_repair"
    # Prefer view_image / Read path successes; drop exploratory shell spam
    if kind.startswith("live"):
        names = []
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                names.append(((tc.get("function") or {}).get("name") or ""))
        final_name = names[-1] if names else ""
        final_blob = _final_tool_blob(msgs)
        path_tools = {"view_image", "read_file", "Read", "read"}
        if final_name in path_tools and _final_tool_is_good_path(msgs):
            return True, "live_path_tool"
        if final_name in ("exec_command", "bash", "Bash") and "LATEST.txt" in final_blob:
            return True, "live_latest_read"
        return False, "live_noise"
    return False, "unclassified"


def _repair_recovers(messages: list) -> bool:
    saw_bad = False
    for m in messages:
        for tc in m.get("tool_calls") or []:
            blob = json.dumps((tc.get("function") or {}).get("arguments"))
            if BAD_PATH_RE.search(blob):
                saw_bad = True
            elif saw_bad and re.search(r"iter-\d{2}\.png", blob):
                return True
    return False


def make_orpo_rejects_from_loops(live_rows: list) -> list:
    """Turn listdir-loop live traces into ORPO pairs vs a gold first-try."""
    pairs = []
    gold = synth_first_try("codex", TASK_SPECS[0])
    for row in live_rows:
        msgs = row.get("messages") or []
        # Count listdir tool turns even on sliced rows (threshold 1 for ORPO reject)
        listdir_turns = []
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                if LISTDIR_RE.search(json.dumps((tc.get("function") or {}).get("arguments"))):
                    listdir_turns.append(m)
                    break
        if len(listdir_turns) < 1 and not is_listdir_loop(msgs):
            continue
        if len(listdir_turns) < 2 and not is_listdir_loop(msgs):
            # only strong rejects: repeated listdir
            continue
        rej_target = listdir_turns[-1]
        # context: messages up to (not including) rejected target, then rejected
        try:
            idx = msgs.index(rej_target)
        except ValueError:
            idx = len(msgs) - 1
        rejected = msgs[:idx] + [rej_target]
        pairs.append(
            {
                "chosen": gold["messages"],
                "rejected": rejected,
                "harness": row.get("harness") or "codex",
                "kind": "orpo_vs_listdir_loop",
                "task": row.get("task"),
                "source_path": row.get("source_path"),
            }
        )
    return pairs


def split_rows(rows: list, seed: int):
    rng = random.Random(seed)
    by_task = {}
    for r in rows:
        by_task.setdefault(r.get("task") or r.get("kind"), []).append(r)
    tasks = list(by_task)
    rng.shuffle(tasks)
    n = len(tasks)
    n_test = max(1, int(0.05 * n))
    n_valid = max(1, int(0.10 * n))
    test_t, valid_t = set(tasks[:n_test]), set(tasks[n_test : n_test + n_valid])
    train, valid, test = [], [], []
    for t, rs in by_task.items():
        bucket = test if t in test_t else valid if t in valid_t else train
        bucket.extend(rs)
    return train, valid, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-first", type=int, default=180)
    ap.add_argument("--n-repair", type=int, default=18)
    ap.add_argument("--n-long", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--codex-glob", default=str(Path.home() / ".codex/sessions/2026/08/*/*.jsonl"))
    ap.add_argument(
        "--pi-glob",
        default=str(Path.home() / ".pi/agent/sessions/--Users-true-Downloads-trace-harvest--/*.jsonl"),
    )
    ap.add_argument(
        "--claude-glob",
        default=str(Path.home() / ".claude/projects/-Users-true-Downloads-trace-harvest/*.jsonl"),
    )
    ap.add_argument("--since-mtime", type=float, default=0, help="unix mtime lower bound for live files")
    ap.add_argument(
        "--harvest-codex",
        default="",
        help="Optional explicit Codex rollout path(s), comma-separated",
    )
    args = ap.parse_args()

    rows = build_synthetic(args.n_first, args.n_repair, args.n_long, args.seed)
    live = []
    orpo_source = []
    codex_paths = list(glob.glob(args.codex_glob))
    if args.harvest_codex:
        codex_paths.extend([p for p in args.harvest_codex.split(",") if p])
    for path in sorted(set(codex_paths)):
        if args.since_mtime and os.path.getmtime(path) < args.since_mtime:
            continue
        live.extend(convert_codex_rollout(path))
        full_msgs = flatten_codex_rollout(path)
        if len(full_msgs) >= 3:
            orpo_source.append(
                {
                    "messages": full_msgs,
                    "harness": "codex",
                    "task": Path(path).stem,
                    "source_path": path,
                }
            )
    for path in glob.glob(args.pi_glob):
        if args.since_mtime and os.path.getmtime(path) < args.since_mtime:
            continue
        live.extend(convert_pi_session(path))
    for path in glob.glob(args.claude_glob):
        if args.since_mtime and os.path.getmtime(path) < args.since_mtime:
            continue
        live.extend(convert_claude_session(path))

    kept, dropped = [], Counter()
    for r in rows + live:
        ok, reason = filter_keep(r)
        if ok:
            r = dict(r)
            r["keep_reason"] = reason
            kept.append(r)
        else:
            dropped[reason] += 1

    orpo = make_orpo_rejects_from_loops(orpo_source + live)
    train, valid, test = split_rows(kept, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, bucket in [("train", train), ("valid", valid), ("test", test)]:
        with open(args.out / f"{name}.jsonl", "w") as f:
            for r in bucket:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out / "orpo_listdir_rejects.jsonl", "w") as f:
        for r in orpo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out / "dropped_live.jsonl", "w") as f:
        for r in live:
            ok, reason = filter_keep(r)
            if not ok:
                f.write(json.dumps({"reason": reason, "task": r.get("task"), "source_path": r.get("source_path")}) + "\n")

    summary = {
        "synthetic": args.n_first + args.n_repair + args.n_long,
        "live_seen": len(live),
        "kept": len(kept),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "orpo_pairs": len(orpo),
        "dropped": dict(dropped),
        "by_kind": dict(Counter(r.get("kind") for r in kept)),
        "by_harness": dict(Counter(r.get("harness") for r in kept)),
        "out": str(args.out),
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
