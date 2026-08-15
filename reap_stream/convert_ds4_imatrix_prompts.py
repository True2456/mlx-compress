"""Convert antirez/ds4's imatrix calibration corpus (gguf-tools/imatrix/dataset/
prompts.jsonl) into our AWQ calibration format.

Why: our own calib/cloud_reap_8k.jsonl has "agentic"/"tool_use" categories but
every record is a single flat prompt -- no multi-turn tool-call trajectories,
so AWQ's salience statistics never see what activations look like several
steps into a real agent loop. ds4's "agent" category has genuine multi-turn
system/user/assistant/tool sequences (up to 17 messages) rendered with
DeepSeek's real chat template and DSML tool-call syntax.

ds4's records carry a "rendered" field: the FULLY chat-template-rendered
string (real special tokens already inserted, e.g.
"<|begin_of_sentence|>..."). awq_quantize_deepseek_v4.py's
_tokenize_prompts() always re-wraps whatever it's given as a single user
turn via apply_chat_template -- that would double-wrap/corrupt an
already-rendered multi-turn string. So each output record here carries
"prerendered": true, and awq_quantize_deepseek_v4.py's loader must skip the
chat-template wrap for those (see the paired patch there).

Usage:
    .venv/bin/python -m reap_stream.convert_ds4_imatrix_prompts \
        --src /path/to/ds4/gguf-tools/imatrix/dataset/prompts.jsonl \
        --out calib/ds4_agentic.jsonl
"""
from __future__ import annotations

import argparse
import json
import random


def convert(src: str, out: str, seed: int) -> None:
    # ds4's prompts.jsonl is clustered by category (all "agent" records
    # first, then "algorithms", then "eval_reasoning", ...). Downstream
    # AWQ calibration just takes the first --n-prompts records in file
    # order (no sampling), so an unshuffled file means any n_prompts <=
    # ~1105 would be 100% agent-category and 0% everything else --
    # shuffle here (seeded, reproducible) so a truncated prefix gets
    # proportional coverage across categories instead.
    records = []
    with open(src) as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rendered = rec.get("rendered")
            if not rendered or not rendered.strip():
                continue
            records.append(
                {
                    "text": rendered,
                    "category": f"ds4_{rec.get('category', 'unknown')}",
                    "prerendered": True,
                    "source": rec.get("source"),
                    "mode": rec.get("mode"),
                }
            )
    random.Random(seed).shuffle(records)
    with open(out, "w") as f_out:
        for rec in records:
            f_out.write(json.dumps(rec) + "\n")
    print(f"[convert-ds4] wrote {len(records)} shuffled prerendered prompts -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    convert(a.src, a.out, a.seed)


if __name__ == "__main__":
    main()
