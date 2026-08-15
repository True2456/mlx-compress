"""Drop rows whose chosen target calls a tool that doesn't exist in Gemma-4's
real deployed tool schema.

Found by checking tool-name overlap between the Step-3.7/DeepSeek-0731
Pi-log harvest and the actual Gemma-4 SFT pool: only bash/edit/read/write
overlap. Everything else (htb_ctf_*, fetch_content, web_search, and a
literal typo'd "fash" -- confirmed present in the raw Pi session log itself,
not an extraction bug: DeepSeek-0731 really did mistype "bash" as "fash" in
that live tool call) is a tool Gemma-4 never has available at inference, so
training bundling behavior against it is wasted signal at best and
reinforces a real tool-calling typo at worst.

Usage:
    .venv/bin/python scripts/filter_tool_schema.py \
        --in data/lora_gemma4/orpo_stopping_pairs_final.jsonl \
        --schema data/lora_gemma4/train.jsonl \
        --out data/lora_gemma4/orpo_stopping_pairs_final_schema.jsonl
"""
from __future__ import annotations

import argparse
import json


def tool_names_from_sft(path):
    names = set()
    for l in open(path):
        r = json.loads(l)
        for m in r.get("messages", []):
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls", []):
                    names.add(tc["function"]["name"])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--schema", default="data/lora_gemma4/train.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    valid_tools = tool_names_from_sft(a.schema)
    print(f"[schema-filter] {len(valid_tools)} valid tool names from {a.schema}", flush=True)

    rows = [json.loads(l) for l in open(a.inp)]
    print(f"[schema-filter] loaded {len(rows)} rows from {a.inp}", flush=True)

    kept, dropped = [], []
    for r in rows:
        target = r["chosen"][-1]
        names = {tc["function"]["name"] for tc in target.get("tool_calls", [])}
        if names <= valid_tools:
            kept.append(r)
        else:
            dropped.append((r, sorted(names - valid_tools)))

    print(f"[schema-filter] kept {len(kept)}, dropped {len(dropped)} (tool not in Gemma-4 schema)", flush=True)

    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"[schema-filter] wrote {len(kept)} -> {a.out}", flush=True)

    dropped_path = a.out.replace(".jsonl", ".dropped_schema.jsonl")
    with open(dropped_path, "w") as f:
        for r, bad_names in dropped:
            f.write(json.dumps({**r, "_bad_tools": bad_names}) + "\n")
    print(f"[schema-filter] wrote {len(dropped)} -> {dropped_path}", flush=True)


if __name__ == "__main__":
    main()
