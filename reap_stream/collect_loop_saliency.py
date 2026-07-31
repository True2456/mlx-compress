"""Layer-wise saliency for choosing a latent-looping recurrent-core boundary.

Different question from collect_gemma4.py's REAP expert saliency (which
measures "how much does removing this hurt output quality"). This measures
"how much does this layer transform the representation, and does that
transformation spike specifically on reasoning-heavy input" -- the metric
the sandwich architecture's loop boundary should actually be chosen from,
per-model, rather than a generic percentage split.

Metric: Block Influence, BI_i = 1 - mean_cos_sim(h_in, h_out) per layer
(ShortGPT-style; scale-invariant, so it isn't confused by layers that just
have larger activation magnitudes). Computed separately over a reasoning-
heavy prompt set and a simple/factual prompt set; the layers where the
differential (reasoning BI - simple BI) is largest are the ones doing
disproportionate reasoning-specific work -- candidate recurrent-core layers.

Reuses collect_gemma4.py's windowed lazy-loading pattern (load lazy, keep
`layers_at_once` decoder blocks resident, stream hidden states through,
free before the next window) so this runs under the same memory ceiling as
the REAP collector already proven on this hardware.

Usage:
    .venv/bin/python -m reap_stream.collect_loop_saliency \
        --model /path/to/gemma-4-12b \
        --out artifacts/loop-saliency-12b
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask

from .collect_gemma4 import _text_model, _tokenize_prompts, _mem_mb, _free_layer

REASONING_PROMPTS = [
    # math / multi-step arithmetic
    "A train leaves city A at 60mph, another leaves city B (300 miles away) "
    "at 40mph toward A one hour later. When and where do they meet?",
    "A tank fills at 5L/min from pipe A and drains at 3L/min from a leak. "
    "Pipe B adds another 2L/min starting at minute 10. If the tank starts "
    "empty and holds 100L, when does it overflow?",
    "Prove by induction that the sum of the first n positive odd integers "
    "equals n squared.",
    "A store marks up cost by 40%, then runs a 25% off sale. If the sale "
    "price is $42, what was the original cost?",
    "Two dice are rolled. What's the probability the sum is prime, given "
    "the sum is greater than 6?",
    "A recipe serves 8 and uses 3/4 cup flour, 2 eggs, and 1.5 tsp salt. "
    "Rescale it for 5 servings, keeping ratios exact.",
    # code tracing / debugging
    "def f(x):\n    if x <= 1:\n        return x\n    return f(x-1) + f(x-2) * 2\n"
    "Trace f(5) step by step and give the final value.",
    "A function processes a queue but items are being dropped under "
    "concurrent load. Walk through the likely race condition step by step "
    "and propose a fix.",
    "This binary search returns wrong results on some inputs:\n"
    "def bsearch(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n"
    "        mid = (lo + hi) // 2\n        if arr[mid] < target: lo = mid\n"
    "        else: hi = mid\n    return lo\nFind the bug and explain why it "
    "causes an infinite loop on some inputs.",
    "A cache implementation evicts the wrong entry under high contention. "
    "Given it uses a naive LRU without locking, trace exactly how two "
    "concurrent reads could corrupt the eviction order.",
    "def memo(f):\n    cache = {}\n    def wrapper(*a):\n        if a not in cache:\n"
    "            cache[a] = f(*a)\n        return cache[a]\n    return wrapper\n"
    "Explain why this breaks for a recursive function decorated with itself, "
    "and what the fix is.",
    # logic puzzles
    "Three people each tell one true and one false statement about who ate "
    "the last cookie. Alice: 'Bob didn't do it. I didn't do it either.' "
    "Bob: 'Carol did it. Alice is lying about something.' Carol: 'I didn't "
    "do it. Bob is telling the truth about something.' Who did it?",
    "If all bloops are razzies, and some razzies are lazzies, but no lazzies "
    "are bloops, is this consistent? Explain the logical structure.",
    "Five houses in a row, each a different color, owner, drink, and pet. "
    "Given: the Brit lives in the red house; the Swede keeps dogs; the Dane "
    "drinks tea; the green house is left of the white house. Who owns the "
    "fish, and what else can you deduce?",
    "A says 'B is lying.' B says 'C is lying.' C says 'A and B are both "
    "lying.' Can all three statements be consistently assigned truth values? "
    "Work through each case.",
    # causal / multi-hop reasoning
    "A server's p99 latency doubled after a deploy that only changed a "
    "logging library. List the plausible causal chains and how you'd test "
    "each one.",
    "Sales dropped 15% the same month a competitor launched and the site "
    "had a redesign. Walk through how you'd isolate which factor (or "
    "combination) actually caused it.",
    "If raising interest rates reduces borrowing, and reduced borrowing "
    "slows construction, and slower construction reduces lumber demand, "
    "what happens to lumber prices two steps removed from a rate hike, and "
    "what could break that chain?",
    # planning / strategy
    "You have 3 tasks with dependencies: B needs A done first, C needs both "
    "A and B done, and A takes 2 hours, B takes 1 hour, C takes 3 hours. "
    "With two workers who can each do one task at a time, what's the "
    "minimum total time to finish all three, and what's the schedule?",
    "Design a rollback plan for a database migration that adds a NOT NULL "
    "column to a 50M-row table with zero downtime allowed. Walk through the "
    "failure modes at each step.",
]

SIMPLE_PROMPTS = [
    # basic facts
    "What is the capital of France?",
    "What year did World War II end?",
    "What is the chemical symbol for gold?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in the solar system?",
    "What language is primarily spoken in Brazil?",
    "How many continents are there?",
    "What is the boiling point of water in Celsius?",
    "What is the currency used in Japan?",
    "Who painted the Mona Lisa?",
    # simple conversions / lookups
    "Convert 10 kilometers to miles.",
    "How many ounces are in a pound?",
    "What is 15% of 200?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "How many days are in a leap year?",
    # single-step instructions
    "Write a function that returns the square of a number.",
    "List three colors.",
    "Write a SQL query to select all rows from a table named users.",
    "Reverse the string 'hello world'.",
    "Write a for loop that prints numbers 1 through 10.",
    "Sort this list in ascending order: [5, 2, 8, 1, 9].",
    "Write a regex that matches a US phone number.",
    "Capitalize the first letter of every word in 'the quick brown fox'.",
]

# Real tools schema (read/edit/search/run) matching what an actual coding
# agent exposes -- close to what this project's own AGENTS.md/Pi harness
# already provides -- so the model has genuine reason to emit a structured
# <tool_call> rather than free text.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string match in a file with new text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the repository for a text pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]

TOOL_CALL_PROMPTS = [
    "The tests in test_parser.py are failing after my last change to "
    "parser.py. Figure out why.",
    "Find every place in the codebase that calls the deprecated "
    "get_user_legacy function.",
    "Check what's actually in config/settings.yaml before we change the "
    "database timeout.",
    "Run the test suite and tell me which tests are failing.",
    "Rename the variable `tmp` to `staging_path` everywhere it's used in "
    "deploy.sh.",
    "Something in the build is broken. Look at the CI log and figure out "
    "the root cause.",
    "Check if there's already a retry decorator somewhere in the utils "
    "module before I write a new one.",
    "The API returns 500 on POST /orders. Reproduce it and find the "
    "stack trace.",
    "Update the version string in package.json to 2.4.0 and confirm it "
    "actually changed.",
    "Find all TODO comments left in the auth module.",
    "Check disk usage on the current directory before we add the new "
    "dataset.",
    "Grep for any hardcoded API keys that shouldn't be committed.",
]

# Neutral, fixed-length padding prepended to every reasoning/simple prompt
# for the length-matched control run. Deliberately inert -- a plausible but
# generic session/environment dump, no reasoning content and no tool-call
# structure -- so it controls for token count without adding a second
# confound of its own. Sized to land near the tool-call category's observed
# ~255-token average once the actual question is appended.
NEUTRAL_PADDING = (
    "Session context: working directory is /home/user/project, a mid-sized "
    "Python web service using FastAPI and PostgreSQL. The repository has "
    "roughly 40,000 lines across 180 files, with a standard layout of "
    "src/, tests/, and config/ directories. Python 3.11 is in use, "
    "dependencies are managed with a lockfile, and the test suite runs "
    "under pytest with coverage reporting enabled. Continuous integration "
    "runs on every push to the main branch and takes about six minutes to "
    "complete. The team follows a standard code review process requiring "
    "at least one approval before merging. Logging is handled through the "
    "standard library logging module, configured to write structured JSON "
    "to stdout in production and human-readable text locally. The service "
    "exposes a small set of REST endpoints and communicates with two "
    "downstream services over HTTP. Deployments happen via a container "
    "image built on every merge to main, pushed to a private registry, and "
    "rolled out with a brief health-check window before traffic shifts "
    "over. Configuration is loaded from environment variables with "
    "sensible local defaults for development. None of this background is "
    "directly relevant to what follows -- it is provided only as ambient "
    "context.\n\n"
)


def _length_matched(prompts: list[str]) -> list[str]:
    return [NEUTRAL_PADDING + p for p in prompts]


def _run_layer(layer, h, window_size):
    """mlx_vlm's DecoderLayer.__call__ returns (h, shared_kv, offset) --
    verified against the actual return statement in gemma4/language.py, not
    its (misleading) -> mx.array type hint. Same 3-tuple shape as mlx_lm's
    equivalent, just built via mlx_vlm's own create_attention_mask since that
    module has its own copy with a different mask-dict convention upstream."""
    if layer.layer_type == "sliding_attention":
        mask = create_attention_mask(h, None, window_size=window_size)
    else:
        mask = create_attention_mask(h, None)
    h_out, _, _ = layer(h, mask=mask, cache=None)
    mx.eval(h_out)
    return h_out


def _block_influence(h_in: mx.array, h_out: mx.array) -> float:
    a = h_in.astype(mx.float32).reshape(-1, h_in.shape[-1])
    b = h_out.astype(mx.float32).reshape(-1, h_out.shape[-1])
    num = (a * b).sum(axis=-1)
    denom = mx.sqrt((a ** 2).sum(axis=-1) + 1e-12) * mx.sqrt((b ** 2).sum(axis=-1) + 1e-12)
    cos = num / (denom + 1e-12)
    mx.eval(cos)
    return float(1.0 - np.array(cos, dtype=np.float64).mean())


def _tokenize_prompts_with_tools(tokenizer, prompts: list[str], max_tokens: int, tools=None) -> list[list[int]]:
    """Same as collect_gemma4._tokenize_prompts but threads a tools schema
    through apply_chat_template -- needed so tool-call prompts actually see
    the tools list and have real reason to emit a structured call."""
    batches: list[list[int]] = []
    for p in prompts:
        try:
            text_in = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text_in = p
        tokens = tokenizer.encode(text_in)[:max_tokens]
        batches.append(tokens)
    return batches


def collect_bi_layerwise(
    model_path: str,
    prompts: list[str],
    max_tokens: int = 256,
    layers_at_once: int = 2,
    tools=None,
) -> tuple[dict[int, float], dict]:
    mx.reset_peak_memory()
    model, processor = load(model_path, lazy=True)
    # mlx_vlm.load returns a processor (image+text), not a plain tokenizer --
    # .encode lives on processor.tokenizer, not the processor itself.
    tokenizer = getattr(processor, "tokenizer", processor)
    text = _text_model(model)
    n_layers = len(text.layers)

    if tools is not None:
        token_batches = _tokenize_prompts_with_tools(tokenizer, prompts, max_tokens, tools=tools)
    else:
        token_batches = _tokenize_prompts(tokenizer, prompts, max_tokens)
    lengths = [len(t) for t in token_batches]
    length_stats = {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }
    print(f"[loop-saliency] prompt token lengths: {length_stats}", flush=True)

    hidden: list[mx.array] = []
    for tokens in token_batches:
        ids = mx.array(tokens)[None]
        h = text.embed_tokens(ids)
        scale = getattr(text, "embed_scale", None)
        if scale is not None:
            h = h * scale
        mx.eval(h)
        hidden.append(h)

    window_size = getattr(text, "window_size", None)
    bi: dict[int, float] = {}

    for window_start in range(0, n_layers, layers_at_once):
        window = list(range(window_start, min(window_start + layers_at_once, n_layers)))
        for layer_idx in window:
            layer = text.layers[layer_idx]
            outs = []
            layer_bi = []
            for h in hidden:
                h_out = _run_layer(layer, h, window_size)
                layer_bi.append(_block_influence(h, h_out))
                outs.append(h_out)
            hidden = outs
            bi[layer_idx] = float(np.mean(layer_bi))
            mem = _mem_mb()
            print(
                f"[loop-saliency] layer {layer_idx:02d}/{n_layers - 1} "
                f"BI={bi[layer_idx]:.4f} active_mb={mem['active_mb']:.0f}",
                flush=True,
            )
        for layer_idx in window:
            _free_layer(text, layer_idx)
        gc.collect()
        mx.clear_cache()

    return bi, length_stats


def _rank_and_print(label: str, bi_a: dict, bi_b: dict) -> dict:
    diff = {i: bi_a[i] - bi_b[i] for i in bi_a}
    ranked = sorted(diff.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nTop 10 layers by {label} Block Influence differential:")
    for i, d in ranked[:10]:
        print(f"  layer {i:2d}: diff={d:+.4f}  (a={bi_a[i]:.4f}, b={bi_b[i]:.4f})")
    return {"differential": diff, "ranked": ranked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--layers-at-once", type=int, default=2)
    ap.add_argument(
        "--length-matched",
        action="store_true",
        help="Prepend neutral padding to reasoning/simple prompts so all "
        "three groups land near the same token length as tool-call prompts "
        "-- isolates whether a differential is content-specific or just "
        "'what happens with a longer prompt'.",
    )
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    reasoning_prompts = _length_matched(REASONING_PROMPTS) if a.length_matched else REASONING_PROMPTS
    simple_prompts = _length_matched(SIMPLE_PROMPTS) if a.length_matched else SIMPLE_PROMPTS

    print("[loop-saliency] reasoning-heavy prompts...", flush=True)
    bi_reasoning, len_reasoning = collect_bi_layerwise(a.model, reasoning_prompts, a.max_tokens, a.layers_at_once)
    print("[loop-saliency] simple/factual prompts...", flush=True)
    bi_simple, len_simple = collect_bi_layerwise(a.model, simple_prompts, a.max_tokens, a.layers_at_once)
    print("[loop-saliency] tool-call prompts...", flush=True)
    bi_tool, len_tool = collect_bi_layerwise(
        a.model, TOOL_CALL_PROMPTS, a.max_tokens, a.layers_at_once, tools=TOOLS_SCHEMA
    )

    reasoning_vs_simple = _rank_and_print("reasoning-vs-simple", bi_reasoning, bi_simple)
    tool_vs_simple = _rank_and_print("tool-call-vs-simple", bi_tool, bi_simple)
    tool_vs_reasoning = _rank_and_print("tool-call-vs-reasoning", bi_tool, bi_reasoning)

    print("\nPrompt token-length stats (length confound check):")
    print(f"  reasoning: {len_reasoning}")
    print(f"  simple:    {len_simple}")
    print(f"  tool_call: {len_tool}")

    report = {
        "bi_reasoning": bi_reasoning,
        "bi_simple": bi_simple,
        "bi_tool_call": bi_tool,
        "token_lengths": {"reasoning": len_reasoning, "simple": len_simple, "tool_call": len_tool},
        "reasoning_vs_simple": reasoning_vs_simple,
        "tool_vs_simple": tool_vs_simple,
        "tool_vs_reasoning": tool_vs_reasoning,
    }
    (out / "loop_saliency_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[loop-saliency] wrote -> {out / 'loop_saliency_report.json'}", flush=True)


if __name__ == "__main__":
    main()
