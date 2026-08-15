"""Run oMLX's own bundled accuracy benchmarks (MMLU/GSM8K/HumanEval/etc.)
against a local checkpoint, without going through the admin HTTP API (which
requires a browser session cookie). Reuses the official benchmark classes
(omlx.eval.*) for dataset loading, prompt formatting, answer extraction, and
scoring -- including HumanEval's sandboxed subprocess test execution -- so
results are directly comparable to what the oMLX app itself would report.
Generation is driven directly via mlx_lm's own BatchGenerator
(mlx_lm.generate.batch_generate), the same continuous-batching engine oMLX's
real server uses, rather than the app's async EngineCore wrapper.

Usage:
    PYTHONPATH=... python -m reap_stream.run_omlx_benchmarks \
        --model ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw-v2 \
        --benchmarks mmlu gsm8k humaneval \
        --n 200 --batch-size 4 \
        --sampling deterministic \
        --out /tmp/bench_v2_deterministic.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


def _load_benchmark(name: str):
    from omlx.eval import BENCHMARKS
    return BENCHMARKS[name]()


async def _score(benchmark, items, texts):
    correct = 0
    category_correct: dict[str, int] = {}
    category_total: dict[str, int] = {}
    per_question = []
    for item, text in zip(items, texts):
        predicted = benchmark.extract_answer(text, item)
        ok = benchmark.check_answer(predicted, item)
        correct += ok
        cat = benchmark.get_category(item)
        if cat:
            category_total[cat] = category_total.get(cat, 0) + 1
            category_correct[cat] = category_correct.get(cat, 0) + ok
        per_question.append({"id": item.get("id"), "correct": ok, "category": cat})
    category_scores = {
        c: category_correct.get(c, 0) / category_total[c] for c in category_total
    }
    return correct, category_scores, per_question


def run_benchmark(model, tokenizer, benchmark_name, n, batch_size, sampler, max_prompt_pad_check=True):
    from omlx.patches.deepseek_v4.chat_template_v4 import apply_chat_template
    from mlx_lm.generate import batch_generate

    benchmark = _load_benchmark(benchmark_name)
    items = asyncio.run(benchmark.load_dataset(sample_size=n))
    print(f"[bench] {benchmark_name}: {len(items)} items loaded", flush=True)

    prompts = []
    for item in items:
        messages = benchmark.format_prompt(item)
        rendered = apply_chat_template(messages, add_generation_prompt=True, thinking_mode="chat")
        prompts.append(tokenizer.encode(rendered))

    max_tokens = benchmark.get_max_tokens()
    t0 = time.time()
    resp = batch_generate(
        model, tokenizer, prompts,
        max_tokens=max_tokens,
        sampler=sampler,
        completion_batch_size=batch_size,
        prefill_batch_size=batch_size,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"[bench] {benchmark_name}: generation done in {elapsed:.1f}s", flush=True)

    correct, category_scores, per_question = asyncio.run(_score(benchmark, items, resp.texts))
    accuracy = correct / len(items) if items else 0.0
    print(f"[bench] {benchmark_name}: accuracy={accuracy:.3f} ({correct}/{len(items)}) "
          f"in {elapsed:.1f}s", flush=True)
    return {
        "benchmark": benchmark_name,
        "n": len(items),
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_sec": elapsed,
        "category_scores": category_scores,
        "per_question": per_question,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sampling", choices=["deterministic", "recommended"], default="deterministic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mtp", action="store_true")
    a = ap.parse_args()

    from omlx.utils.model_loading import load_text_model
    from omlx.model_settings import ModelSettings
    from mlx_lm.sample_utils import make_sampler

    if a.sampling == "deterministic":
        sampler = make_sampler(temp=0.0)
    else:
        # DeepSeek-V4-Flash-0731's own README-recommended settings for
        # agentic scenarios: temperature=1.0, top_p=0.95.
        sampler = make_sampler(temp=1.0, top_p=0.95)

    print(f"[bench] loading model: {a.model} (mtp_enabled={a.mtp})", flush=True)
    model, tokenizer = load_text_model(a.model, model_settings=ModelSettings(mtp_enabled=a.mtp))

    results = []
    for name in a.benchmarks:
        results.append(run_benchmark(model, tokenizer, name, a.n, a.batch_size, sampler))

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": a.model, "sampling": a.sampling, "results": results}, indent=2))
    print(f"[bench] wrote results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
