"""Build the DWQ phase-1 target-collection dataset from ds4's corpus, with the
"agent" category overweighted relative to its natural share.

Why: the only existing DWQ targets (artifacts/dwq-targets-dsv4) were
collected from calib/cloud_reap_8k.jsonl -- the old, single-turn-only
calibration set, superseded for the same reason AWQ moved off it (see
[[deepseek-v4-agentic-looping]] memory). Re-collecting from calib/ds4_agentic.jsonl
alone would only give agent its natural 23.6% share; since DWQ's whole point
here is fixing agentic-context repetition, this oversamples it while keeping
every other category's *relative* proportions intact (source stays dominant
among non-agent, etc.) so general-capability signal isn't discarded.

Takes ALL unique agent-category records, then samples non-agent records so
agent lands at --agent-share of the final set (default 0.45, up from the
corpus's natural 23.6%).

Usage:
    python3 -m reap_stream.build_dwq_target_dataset \
        --src calib/ds4_agentic.jsonl \
        --out calib/dwq_targets_agentic_weighted.jsonl \
        --agent-share 0.45 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random


def build(src: str, out: str, agent_share: float, seed: int) -> None:
    agent, non_agent = [], []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            (agent if rec.get("category") == "ds4_agent" else non_agent).append(rec)

    rng = random.Random(seed)
    rng.shuffle(non_agent)

    n_agent = len(agent)
    # agent_share = n_agent / (n_agent + n_non_agent_kept)
    n_non_agent = int(round(n_agent * (1 - agent_share) / agent_share))
    n_non_agent = min(n_non_agent, len(non_agent))
    kept_non_agent = non_agent[:n_non_agent]

    combined = agent + kept_non_agent
    rng.shuffle(combined)

    with open(out, "w") as f_out:
        for rec in combined:
            f_out.write(json.dumps(rec) + "\n")

    import collections
    cats = collections.Counter(r["category"] for r in combined)
    print(f"[build-dwq-targets] wrote {len(combined)} records -> {out}")
    print(f"[build-dwq-targets] agent share: {n_agent}/{len(combined)} = {n_agent/len(combined):.1%}")
    print(f"[build-dwq-targets] category breakdown: {cats.most_common()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent-share", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.src, a.out, a.agent_share, a.seed)


if __name__ == "__main__":
    main()
