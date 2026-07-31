"""HumanEval pass@1 for the Gemma-4-12B agentic LoRA: base vs adapted.

Real functional benchmark, not just loss/PPL -- this project has already
been burned once by trusting PPL alone (the REAM merge scheme looked like a
win on PPL but was flat on actual accuracy). Execute each generated solution
against HumanEval's real test cases; pass/fail, not a proxy.

The model was fine-tuned as a chat-formatted agentic coding assistant (system
prompt "You are an autonomous coding agent...", responses in ```python
fences), not a raw base/completion model -- so problems are wrapped in a user
turn and the code block is extracted from the response, matching how it was
actually trained to be used, rather than feeding the bare HumanEval prompt as
a raw continuation.

Usage:
    .venv/bin/python scripts/eval_gemma4_humaneval.py \
        --model ~/.lmstudio/models/mlx-community/gemma-4-12B-it-qat-4bit \
        --adapter-path adapters/gemma4-12b-agentic \
        --out artifacts/humaneval_gemma4_12b_lora.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile

from datasets import load_dataset
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

SYSTEM_PROMPT = (
    "You are an autonomous coding agent running on the user's machine. "
    "Write correct, complete Python code."
)


def extract_code(text, entry_point):
    fences = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    for block in fences:
        if f"def {entry_point}" in block:
            return block
    if fences:
        return fences[0]
    return text


def run_test(code, test, entry_point, timeout=10):
    program = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, path], capture_output=True, timeout=timeout, text=True
        )
        return r.returncode == 0, (r.stderr[-500:] if r.returncode != 0 else "")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--n", type=int, default=164)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"[eval] loading {a.model} (adapter={a.adapter_path})", flush=True)
    model, processor = load(
        a.model, adapter_path=a.adapter_path, processor_config={"trust_remote_code": True}
    )
    config = model.config.__dict__

    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = list(ds)[: a.n]

    results = []
    n_pass = 0
    for i, p in enumerate(problems):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Complete the following Python function. Respond with the "
                    "complete function (signature + body) in a single ```python "
                    "code block, nothing else.\n\n" + p["prompt"]
                ),
            },
        ]
        prompt = apply_chat_template(processor, config, messages, add_generation_prompt=True)
        out = generate(model, processor, prompt, image=None, max_tokens=a.max_tokens, verbose=False)
        text = out.text if hasattr(out, "text") else str(out)
        code = extract_code(text, p["entry_point"])
        ok, err = run_test(code, p["test"], p["entry_point"])
        n_pass += ok
        results.append({"task_id": p["task_id"], "pass": ok, "error": err if not ok else ""})
        print(f"  [{i+1}/{len(problems)}] {p['task_id']}: {'PASS' if ok else 'FAIL'}", flush=True)

    summary = {
        "model": a.model,
        "adapter_path": a.adapter_path,
        "n": len(problems),
        "n_pass": n_pass,
        "pass_at_1": n_pass / len(problems),
        "results": results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=2)
        print(f"[eval] wrote -> {a.out}")


if __name__ == "__main__":
    main()
