#!/usr/bin/env python3
"""Export live Codex/Pi/Claude sessions into unified curriculum packs.

Only live harness sessions — no foreign dataset imports.
Writes full unsliced episodes + cumulative next-assistant slices.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Reuse converters from path harvest script
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load_harvest():
    path = HERE / "harvest_harness_path_traces.py"
    spec = importlib.util.spec_from_file_location("harvest_harness_path_traces", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def tag_capabilities(messages: list, brief: str = "") -> list[str]:
    caps = set()
    blob = json.dumps(messages) + "\n" + brief
    if re.search(r"view_image|Read|read_file|screenshot|\.png|\.jpeg", blob, re.I):
        # only count if an image tool was actually used
        for m in messages:
            for tc in m.get("tool_calls") or []:
                name = ((tc.get("function") or {}).get("name") or "")
                args = json.dumps((tc.get("function") or {}).get("arguments"))
                if name in ("view_image", "Read", "read", "read_file") and re.search(
                    r"\.(png|jpe?g|webp|gif)", args, re.I
                ):
                    caps.add("multimodal_vision")
                if "webgpu" in args.lower() or "three" in blob.lower():
                    caps.add("visual_prog_webgpu")
    if "webgpu" in blob.lower() or "WebGPU" in blob:
        caps.add("visual_prog_webgpu")
    if re.search(r"ui_demo|screenshot|header|contrast|WCAG", blob):
        caps.add("ui_inspect")
    if re.search(r"\b\d+\s*\*\s*\d+|\b\d+\s*\+\s*\d+|pixel_count|outer_width|960|720", blob):
        caps.add("gsm8k_style")
    if re.search(r"sRGB|color space|pytest|assertion|contrast|linear workflow|WCAG", blob, re.I):
        caps.add("mmlu_style")
    asst_tools = sum(
        1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls")
    )
    if asst_tools >= 6:
        caps.add("agentic_long")
    elif asst_tools >= 3:
        caps.add("agentic_medium")
    return sorted(caps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "lora_step37_unified")
    ap.add_argument("--codex", action="append", default=[])
    ap.add_argument("--pi", action="append", default=[])
    ap.add_argument("--claude", action="append", default=[])
    ap.add_argument("--brief", action="append", default=[], help="brief.md path aligned by order")
    args = ap.parse_args()
    h = _load_harvest()

    episodes = []
    for i, path in enumerate(args.codex):
        brief = Path(args.brief[i]).read_text() if i < len(args.brief) else ""
        rows = h.convert_codex_rollout(path)
        if not rows:
            continue
        full = max(rows, key=lambda r: len(r["messages"]))
        full = dict(full)
        full["kind"] = "live_episode"
        full["brief"] = brief[:2000]
        full["capabilities"] = tag_capabilities(full["messages"], brief)
        full["source_path"] = path
        # attach all slices
        full["_slices"] = rows
        episodes.append(full)

    for path in args.pi:
        rows = h.convert_pi_session(path)
        if not rows:
            continue
        full = max(rows, key=lambda r: len(r["messages"]))
        full = dict(full)
        full["kind"] = "live_episode"
        full["capabilities"] = tag_capabilities(full["messages"])
        full["source_path"] = path
        full["_slices"] = rows
        episodes.append(full)

    for path in args.claude:
        rows = h.convert_claude_session(path)
        if not rows:
            continue
        full = max(rows, key=lambda r: len(r["messages"]))
        full = dict(full)
        full["kind"] = "live_episode"
        full["capabilities"] = tag_capabilities(full["messages"])
        full["source_path"] = path
        full["_slices"] = rows
        episodes.append(full)

    cand_dir = args.out / "candidates"
    full_dir = args.out / "full"
    sliced_dir = args.out / "sliced"
    for d in (cand_dir, full_dir, sliced_dir):
        d.mkdir(parents=True, exist_ok=True)

    kept_full, kept_slices = [], []
    dropped = Counter()
    for ep in episodes:
        msgs = ep["messages"]
        if h.is_listdir_loop(msgs):
            dropped["listdir_loop"] += 1
            continue
        caps = ep.get("capabilities") or []
        asst = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
        # Accept medium+ for this first harvest batch; prefer long when available
        if asst < 3:
            dropped["too_short"] += 1
            continue
        if len(caps) < 2:
            dropped["few_capabilities"] += 1
            # still keep as candidate for review
            with open(cand_dir / "needs_review.jsonl", "a") as f:
                f.write(json.dumps({k: v for k, v in ep.items() if k != "_slices"}) + "\n")
            continue
        row = {k: v for k, v in ep.items() if k != "_slices"}
        kept_full.append(row)
        for sl in ep.get("_slices") or []:
            ok, reason = h.filter_keep(sl)
            if ok or sl.get("kind") in ("live_first_try", "live_repair", "live_rollout"):
                # keep path-good slices; skip listdir
                if h.is_listdir_loop(sl.get("messages") or []):
                    continue
                if h.BAD_PATH_RE.search(h._final_tool_blob(sl.get("messages") or [])):
                    continue
                s = dict(sl)
                s["capabilities"] = caps
                s["parent_source"] = ep.get("source_path")
                kept_slices.append(s)

    def write_split(dir_path: Path, rows: list):
        # tiny first batch: all train except last as valid if >=2
        if len(rows) >= 2:
            train, valid = rows[:-1], rows[-1:]
        else:
            train, valid = rows, []
        for name, bucket in [("train", train), ("valid", valid), ("test", [])]:
            with open(dir_path / f"{name}.jsonl", "w") as f:
                for r in bucket:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_split(full_dir, kept_full)
    write_split(sliced_dir, kept_slices)
    with open(cand_dir / "all_episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps({k: v for k, v in ep.items() if k != "_slices"}, ensure_ascii=False) + "\n")

    summary = {
        "episodes_seen": len(episodes),
        "full_kept": len(kept_full),
        "slices_kept": len(kept_slices),
        "dropped": dict(dropped),
        "capability_hist": dict(Counter(c for ep in kept_full for c in ep.get("capabilities", []))),
        "sources": [ep.get("source_path") for ep in kept_full],
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
