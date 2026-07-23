"""Middle-canvas interventions for Step-3.7 (Painters / looping diagnostics).

Runs on a frozen MLX checkpoint without writing weights. Compares teacher-forced
NLL under: baseline, skip-middle, reverse-middle, middle-repeat — streaming one
decoder layer at a time so peak RAM stays ~one MoE block.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable, Optional

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask

from .dataset import load_prompt_texts


DEFAULT_MODEL = str(
    Path.home() / ".lmstudio/models/mlx-community/Step-3.7-Flash-148B-MLX"
)


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _lm_head(model):
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "lm_head"):
        return lm.lm_head
    raise RuntimeError("language_model.lm_head not found")


def _mem_mb() -> dict[str, float]:
    return {
        "active_mb": mx.get_active_memory() / (1024**2),
        "peak_mb": mx.get_peak_memory() / (1024**2),
    }


def _tokenize(processor, prompts: list[str], max_tokens: int) -> list[list[int]]:
    tok = getattr(processor, "tokenizer", processor)
    out: list[list[int]] = []
    for p in prompts:
        if hasattr(tok, "apply_chat_template"):
            try:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = p
        else:
            text = p
        ids = tok.encode(text)[:max_tokens]
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if len(ids) < 4:
            continue
        out.append(list(ids))
    return out


def _run_layer(layer, h, sliding_window: int | None):
    if getattr(layer, "is_sliding", False):
        mask = create_attention_mask(h, None, window_size=sliding_window)
    else:
        mask = create_attention_mask(h, None)
    h_out = layer(h, mask=mask, cache=None)
    mx.eval(h_out)
    return h_out


def build_schedules(
    n_layers: int,
    middle_start: int,
    middle_end: int,
    skip_stride: int = 2,
    repeat_layer: int | None = None,
    repeat_times: int = 3,
) -> dict[str, list[int]]:
    """Build layer-index schedules. Indices are into text.layers."""
    if not (0 <= middle_start < middle_end < n_layers):
        raise ValueError(f"bad middle band [{middle_start},{middle_end}) vs n={n_layers}")

    baseline = list(range(n_layers))

    skip: list[int] = []
    for i in range(n_layers):
        if middle_start <= i < middle_end and ((i - middle_start) % skip_stride == 1):
            continue
        skip.append(i)

    reverse = (
        list(range(0, middle_start))
        + list(range(middle_end - 1, middle_start - 1, -1))
        + list(range(middle_end, n_layers))
    )

    rep_at = repeat_layer if repeat_layer is not None else (middle_start + middle_end) // 2
    if not (middle_start <= rep_at < middle_end):
        raise ValueError(f"repeat_layer {rep_at} outside middle band")
    repeat: list[int] = []
    for i in range(n_layers):
        if i == rep_at:
            repeat.extend([i] * repeat_times)
        else:
            repeat.append(i)

    return {
        "baseline": baseline,
        "skip_middle": skip,
        "reverse_middle": reverse,
        "middle_repeat": repeat,
    }


def _nll_for_schedule(
    text,
    lm_head,
    token_batches: list[list[int]],
    schedule: list[int],
    sliding_window: int | None,
) -> dict:
    """Stream layers in schedule order; mean token NLL over batches."""
    total_nll = 0.0
    total_tok = 0
    peak = 0.0

    for tokens in token_batches:
        ids = mx.array(tokens)[None]
        h = text.embed_tokens(ids)
        mx.eval(h)
        for layer_idx in schedule:
            h = _run_layer(text.layers[layer_idx], h, sliding_window)
            peak = max(peak, mx.get_peak_memory() / (1024**2))

        h = text.norm(h)
        logits = lm_head(h)
        mx.eval(logits)
        # teacher-forced NLL on positions 0..T-2 predicting 1..T-1
        log_probs = logits.astype(mx.float32) - mx.logsumexp(
            logits.astype(mx.float32), axis=-1, keepdims=True
        )
        target = mx.array(tokens[1:])[None]
        gathered = mx.take_along_axis(log_probs[:, :-1, :], target[..., None], axis=-1)
        gathered = gathered.squeeze(-1)
        mx.eval(gathered)
        nll = float((-gathered).sum().item())
        total_nll += nll
        total_tok += len(tokens) - 1
        del h, logits, log_probs, gathered
        gc.collect()
        mx.clear_cache()

    return {
        "mean_nll": total_nll / max(total_tok, 1),
        "tokens": total_tok,
        "peak_mb": peak,
        "schedule_len": len(schedule),
        "unique_layers": len(set(schedule)),
    }


def run_interventions(
    model_path: str,
    output_dir: str | Path,
    middle_start: int = 12,
    middle_end: int = 36,
    skip_stride: int = 2,
    repeat_times: int = 3,
    max_tokens: int = 96,
    max_samples: int = 16,
    dataset_file: Optional[str] = None,
    variants: Optional[Iterable[str]] = None,
) -> dict:
    mx.reset_peak_memory()
    model, processor = load(model_path, lazy=True)
    text = _text_model(model)
    lm_head = _lm_head(model)
    cfg = getattr(text, "args", None)
    sliding_window = getattr(cfg, "sliding_window", None) if cfg else None
    n_layers = len(text.layers)

    if dataset_file:
        prompts = load_prompt_texts(dataset_file, limit=max_samples)
    else:
        prompts = [
            "Write a Python function that merges two sorted lists.",
            "Prove that the sum of the first n odd numbers equals n squared.",
            "What is 17 * 19? Show steps.",
            "Explain mixture-of-experts routing in one paragraph.",
            "def binary_search(arr, x):\n",
            "List three failure modes of tool-using agents.",
            "Solve: if a=3, b=a, c=b+2, what is c?",
            "Summarize residual connections in transformers.",
        ][:max_samples]

    batches = _tokenize(processor, prompts, max_tokens)
    schedules = build_schedules(
        n_layers,
        middle_start=middle_start,
        middle_end=middle_end,
        skip_stride=skip_stride,
        repeat_times=repeat_times,
    )
    if variants is not None:
        schedules = {k: v for k, v in schedules.items() if k in set(variants)}

    results: dict = {
        "model": model_path,
        "n_layers": n_layers,
        "middle_band": [middle_start, middle_end],
        "max_tokens": max_tokens,
        "n_prompts": len(batches),
        "variants": {},
        "load_mem": _mem_mb(),
    }

    # Drop the shared model — each variant reloads so peak stays one full pass.
    del model, processor, text, lm_head
    gc.collect()
    mx.clear_cache()

    baseline_nll = None
    for name, sched in schedules.items():
        print(f"== {name} (len={len(sched)}) ==", flush=True)
        mx.reset_peak_memory()
        model, processor = load(model_path, lazy=True)
        text = _text_model(model)
        lm_head = _lm_head(model)
        cfg = getattr(text, "args", None)
        sliding_window = getattr(cfg, "sliding_window", None) if cfg else None
        stats = _nll_for_schedule(text, lm_head, batches, sched, sliding_window)
        del model, processor, text, lm_head
        gc.collect()
        mx.clear_cache()
        if name == "baseline":
            baseline_nll = stats["mean_nll"]
        delta = None if baseline_nll is None else stats["mean_nll"] - baseline_nll
        row = {**stats, "delta_nll_vs_baseline": delta}
        results["variants"][name] = row
        print(
            f"  mean_nll={stats['mean_nll']:.4f} "
            f"delta={delta if delta is None else f'{delta:+.4f}'} "
            f"peak_mb={stats['peak_mb']:.0f}",
            flush=True,
        )

    # Painters-style interpretation hints
    v = results["variants"]
    interpretation = []
    if "baseline" in v and "skip_middle" in v:
        interpretation.append(
            "skip_ok"
            if v["skip_middle"]["delta_nll_vs_baseline"] < 0.15
            else "skip_hurts"
        )
    if "baseline" in v and "middle_repeat" in v and "skip_middle" in v:
        if (
            v["middle_repeat"]["delta_nll_vs_baseline"]
            > v["skip_middle"]["delta_nll_vs_baseline"] + 0.05
        ):
            interpretation.append("repeat_worse_than_skip_as_painters_predicts")
        else:
            interpretation.append("repeat_not_worse_than_skip")
    if "baseline" in v and "reverse_middle" in v:
        interpretation.append(
            "order_matters"
            if v["reverse_middle"]["delta_nll_vs_baseline"] > 0.05
            else "order_robust"
        )
    results["interpretation"] = interpretation

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "interventions.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {path}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Step-3.7 middle-canvas interventions")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default="artifacts/step37-interventions")
    p.add_argument("--middle-start", type=int, default=12)
    p.add_argument("--middle-end", type=int, default=36)
    p.add_argument("--skip-stride", type=int, default=2)
    p.add_argument("--repeat-times", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--max-samples", type=int, default=16)
    p.add_argument("--dataset-file", default=None)
    p.add_argument(
        "--variants",
        default="baseline,skip_middle,reverse_middle,middle_repeat",
        help="Comma-separated subset of interventions",
    )
    args = p.parse_args()
    run_interventions(
        model_path=args.model,
        output_dir=args.output,
        middle_start=args.middle_start,
        middle_end=args.middle_end,
        skip_stride=args.skip_stride,
        repeat_times=args.repeat_times,
        max_tokens=args.max_tokens,
        max_samples=args.max_samples,
        dataset_file=args.dataset_file,
        variants=[v.strip() for v in args.variants.split(",") if v.strip()],
    )


if __name__ == "__main__":
    main()
