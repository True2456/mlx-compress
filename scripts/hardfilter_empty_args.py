"""Deterministic post-filter on top of the model-judge pass: hard-drop any
row whose chosen target tool call has empty arguments, UNLESS that tool is
observed to be legitimately no-argument everywhere else in the dataset
(e.g. *_profile lookup tools, which are 100% empty-args across every
instance and are the real thing, not a defect).

The judge (scripts/judge_orpo_pairs.py) caught most empty-args rows but
missed 87 of them (verified: data/lora_gemma4/orpo_stopping_pairs_judged.jsonl
has empty-args tool calls in both kept and dropped rows). This is a
deterministic code check, not a model judgment call, so it catches 100% of
the cases the judge's non-determinism let through.

A tool is classified "legitimately no-arg" only if EVERY occurrence of that
tool name across the full judged set has empty args -- if it's ever seen
with real args, an empty-args instance for that same tool is a defect.

Usage:
    .venv/bin/python scripts/hardfilter_empty_args.py \
        --in data/lora_gemma4/orpo_stopping_pairs_judged.kept.jsonl \
        --out data/lora_gemma4/orpo_stopping_pairs_final.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def is_empty(args):
    return args in ({}, None, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--reference",
        default="data/lora_gemma4/orpo_stopping_pairs_judged.jsonl",
        help="full judged set (kept+dropped) used to classify which tools are legitimately no-arg",
    )
    ap.add_argument(
        "--min-samples", type=int, default=3,
        help="minimum occurrences of a tool before 'always empty' is trusted as legitimate no-arg; "
             "below this, a single data point can't establish legitimacy (e.g. WebFetch/WebSearch "
             "with n=1 each -- a search/fetch obviously needs a query/url, so under-sampled 'always "
             "empty' tools are treated as suspect, not legitimate)",
    )
    a = ap.parse_args()

    ref_rows = [json.loads(l) for l in open(a.reference)]
    tool_stats = defaultdict(lambda: [0, 0])  # name -> [total, empty]
    for r in ref_rows:
        for tc in r["chosen"][-1].get("tool_calls", []):
            name = tc["function"]["name"]
            tool_stats[name][0] += 1
            if is_empty(tc["function"]["arguments"]):
                tool_stats[name][1] += 1

    legit_no_arg = {
        name for name, (total, empty) in tool_stats.items()
        if total == empty and total >= a.min_samples
    }
    under_sampled_empty = {
        name for name, (total, empty) in tool_stats.items()
        if total == empty and total < a.min_samples
    }
    # Naming-convention override: *_profile lookup tools and
    # memory_retrieve_summary are a confirmed no-arg tool family (checked
    # against the full source pool, not just this judged subset -- they
    # never appear with non-empty args anywhere). Under-sampled members of
    # that same family are trusted too; WebFetch/WebSearch are NOT part of
    # this family and stay suspect (a search/fetch needs a query/url).
    family_override = {
        n for n in under_sampled_empty
        if n.endswith("_profile") or n == "memory_retrieve_summary"
    }
    if family_override:
        legit_no_arg |= family_override
        under_sampled_empty -= family_override
    print(f"[hardfilter] tools classified legitimately no-arg (always empty, n>={a.min_samples}, or *_profile family): {sorted(legit_no_arg)}", flush=True)
    if under_sampled_empty:
        print(f"[hardfilter] under-sampled always-empty tools treated as SUSPECT, not legit: {sorted(under_sampled_empty)}", flush=True)

    rows = [json.loads(l) for l in open(a.inp)]
    print(f"[hardfilter] loaded {len(rows)} rows from {a.inp}", flush=True)

    kept, dropped = [], []
    for r in rows:
        target = r["chosen"][-1]
        bad = False
        for tc in target.get("tool_calls", []):
            name = tc["function"]["name"]
            if name in legit_no_arg:
                continue
            if is_empty(tc["function"]["arguments"]):
                bad = True
                break
        (dropped if bad else kept).append(r)

    print(f"[hardfilter] kept {len(kept)}, dropped {len(dropped)} (empty required-args)", flush=True)

    out = {k: v for k, v in {"kept": a.out}.items()}
    with open(a.out, "w") as f:
        for r in kept:
            r2 = {k: v for k, v in r.items() if k != "_judge"}
            f.write(json.dumps(r2) + "\n")
    print(f"[hardfilter] wrote {len(kept)} -> {a.out}", flush=True)

    dropped_path = a.out.replace(".jsonl", ".dropped_empty_args.jsonl")
    with open(dropped_path, "w") as f:
        for r in dropped:
            f.write(json.dumps(r) + "\n")
    print(f"[hardfilter] wrote {len(dropped)} -> {dropped_path}", flush=True)


if __name__ == "__main__":
    main()
