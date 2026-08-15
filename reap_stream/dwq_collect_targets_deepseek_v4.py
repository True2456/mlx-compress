"""DWQ phase 1 for DeepSeek-V4-Flash: stream the native (mxfp4/mxfp8) teacher
block-by-block and cache top-k logit targets for each calibration prompt.

Port of dwq_collect_targets.py's streaming design, using
collect_deepseek_v4.py's verified-correct per-layer calling convention
instead of Step-3.7's -- three real architecture differences that break a
naive reuse of the Step-3.7 collector:

1. Hyper-Connections: hidden state is (batch, seq, hc_mult, hidden), not
   (batch, seq, hidden) -- built once at the embedding stage via
   mx.broadcast_to + mx.contiguous (matching DeepseekV4Model.__call__
   exactly), carried through every layer, and only collapsed back to
   (batch, seq, hidden) at the very end via hc_head.
2. Hash-routed layers (first num_hash_layers) need input_ids threaded into
   every layer call for MoEGate's fixed token->expert lookup -- layer
   signature is `layer(h, mask, cache, input_ids)`, not `layer(h, mask=,
   cache=)`.
3. Final projection is `lm_head(norm(hc_head(h)))`, not `lm_head(norm(h))`
   -- verified against LanguageModel.__call__ / DeepseekV4Model.__call__ in
   the installed mlx_vlm source line-for-line, not assumed.

Usage:
    .venv/bin/python -m reap_stream.dwq_collect_targets_deepseek_v4 \
        --teacher ~/Desktop/models/DeepSeek-V4-Flash-0731 \
        --dataset calib/cloud_reap_8k.jsonl \
        --out artifacts/dwq-targets-dsv4 \
        --n-prompts 2048 --max-tokens 384 --topk 128 --layers-at-once 1 \
        --exclude-categories multimodal
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _load_filtered_prompts(
    dataset_file: str, limit: int, exclude_categories: set[str]
) -> list[tuple[str, bool]]:
    """Returns (text, prerendered) pairs. prerendered=True means text is
    already a fully chat-template-rendered string (real special tokens
    included) and must NOT be wrapped again via apply_chat_template -- see
    reap_stream/awq_quantize_deepseek_v4.py's identical fix; this script had
    the same bug (always wrapping as a single user turn), which would
    double-wrap/corrupt calib/ds4_agentic.jsonl's prerendered multi-turn
    agent trajectories."""
    out = []
    with open(dataset_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("category") in exclude_categories:
                continue
            text = rec.get("text")
            if text and str(text).strip():
                out.append((str(text).strip(), bool(rec.get("prerendered", False))))
            if len(out) >= limit:
                break
    if not out:
        raise ValueError(f"no usable prompts in {dataset_file} after excluding {exclude_categories}")
    return out


def _tokenize_prompts(
    tokenizer, prompts: list[tuple[str, bool]], max_tokens: int
) -> list[list[int]]:
    batches = []
    for p, prerendered in prompts:
        if prerendered:
            text_in = p
        elif hasattr(tokenizer, "apply_chat_template"):
            try:
                text_in = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text_in = p
        else:
            text_in = p
        tokens = tokenizer.encode(text_in)[:max_tokens]
        batches.append(tokens)
    return batches


def _expand_hc(h: mx.array, hc_mult: int) -> mx.array:
    h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], hc_mult, h.shape[2]))
    return mx.contiguous(h)


class _DropLayer(nn.Module):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("freed layer was invoked -- streaming collector bug")


def _free_layer(text, layer_idx: int) -> None:
    text.layers[layer_idx] = _DropLayer()


def _run_layer(layer, h, mask, input_ids):
    h_out = layer(h, mask, None, input_ids)
    mx.eval(h_out)
    return h_out


def collect_targets(
    teacher_path: str,
    dataset_file: str,
    out_dir: str,
    n_prompts: int,
    max_tokens: int,
    topk: int,
    layers_at_once: int,
    exclude_categories: list[str],
    force_ids=(1,),
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[dwq-collect-dsv4] loading teacher (lazy): {teacher_path}", flush=True)
    model, processor = load(teacher_path, lazy=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    lm = model.language_model
    text = _text_model(model)
    n_layers = len(text.layers)
    hc_mult = text.args.hc_mult
    sliding_window = text.args.sliding_window

    prompts = _load_filtered_prompts(dataset_file, n_prompts, set(exclude_categories))
    token_batches = _tokenize_prompts(tokenizer, prompts, max_tokens)
    print(f"[dwq-collect-dsv4] {len(token_batches)} prompts (excluded {exclude_categories}), "
          f"{n_layers} layers, hc_mult={hc_mult}, topk={topk}, "
          f"layers_at_once={layers_at_once}", flush=True)

    t0 = time.time()
    hidden, input_ids_list, masks = [], [], []
    for tokens in token_batches:
        ids = mx.array(tokens)[None]
        h = _expand_hc(text.embed_tokens(ids), hc_mult)
        mask = create_attention_mask(h[:, :, 0, :], None, window_size=sliding_window, return_array=True)
        mx.eval(h, mask)
        hidden.append(h)
        input_ids_list.append(ids)
        masks.append(mask)

    for w0 in range(0, n_layers, layers_at_once):
        window = range(w0, min(w0 + layers_at_once, n_layers))
        for li in window:
            layer = text.layers[li]
            hidden = [
                _run_layer(layer, h, mask, ids)
                for h, mask, ids in zip(hidden, masks, input_ids_list)
            ]
        for li in window:
            _free_layer(text, li)
        gc.collect()
        mx.clear_cache()
        act = mx.get_active_memory() / 1e9
        print(f"[dwq-collect-dsv4] blocks {w0}-{window[-1]}/{n_layers - 1} "
              f"active={act:.1f} GB ({time.time() - t0:.0f}s elapsed)", flush=True)

    print("[dwq-collect-dsv4] final hc_head + norm + lm_head -> top-k targets", flush=True)
    for i in range(len(hidden)):
        collapsed = text.norm(text.hc_head(hidden[i]))
        logits = lm.lm_head(collapsed)[0]
        vals, idx = mx.top_k_with_indices(logits, topk, axis=-1) \
            if hasattr(mx, "top_k_with_indices") else _topk(logits, topk)

        # Force decision-critical tokens into the retained support at EVERY
        # position, whatever their rank. This is "decision-critical support
        # omission" (arXiv 2607.07050): a teacher's top-k can hold 99.99% of
        # probability mass yet omit the token that decides a behaviour, and
        # omitted tokens receive ZERO gradient, so the student inflates them
        # for free. Measured here with topk=128: EOS was present at only 70%
        # of positions, and the student drove P(EOS) from 0.0004 to 0.33 at
        # agentic turn-starts -- it stopped generating entirely.
        #
        # Note this is a SUPPORT problem, not a mass problem: the missing mass
        # is ~6e-4, so any penalty scaled by it (e.g. a tail/rest bucket) is
        # far too weak to fix this. The token has to be in the set.
        # NOTE: each forced token gets its OWN slot. Writing them all into the
        # last column instead lets a later token evict an earlier one wherever
        # both are absent -- measured: that gave </think> 100% coverage but EOS
        # only 92.53%.
        for slot, tid in enumerate(force_ids or ()):
            col = idx.shape[-1] - 1 - slot
            present = mx.any(idx == tid, axis=-1)
            # Drop the lowest-ranked entries in favour of the forced tokens
            # where they are missing; at k=1024 those entries are negligible.
            new_idx = mx.where(present, idx[:, col], tid)[:, None]
            new_val = mx.where(present, vals[:, col], logits[:, tid])[:, None]
            idx = mx.concatenate([idx[:, :col], new_idx, idx[:, col + 1:]], axis=-1)
            vals = mx.concatenate([vals[:, :col], new_val, vals[:, col + 1:]], axis=-1)
        # Full-vocab log-partition, so training can constrain the probability
        # mass OUTSIDE the top-k. Without it the loss only sees these `topk`
        # tokens and the student is free to inflate every other logit at no
        # cost -- measured: out-of-top-k mass +84% and P(EOS) x63 after 100
        # steps, which promoted <|end_of_sentence|> to the top-1 token in
        # multi-turn agentic context and killed generation entirely.
        logz = mx.logsumexp(logits.astype(mx.float32), axis=-1)
        mx.eval(vals, idx, logz)
        mx.save_safetensors(
            str(out / f"{i:05d}.safetensors"),
            {
                "input_ids": input_ids_list[i][0].astype(mx.int32),
                "topk_vals": vals.astype(mx.float16),
                "topk_idx": idx.astype(mx.int32),
                "logz": logz.astype(mx.float32),
            },
        )
        hidden[i] = None
        if (i + 1) % 100 == 0:
            print(f"[dwq-collect-dsv4] wrote {i + 1}/{len(hidden)}", flush=True)

    meta = {
        "teacher": teacher_path,
        "dataset": dataset_file,
        "excluded_categories": exclude_categories,
        "n_prompts": len(input_ids_list),
        "max_tokens": max_tokens,
        "topk": topk,
        "force_ids": list(force_ids or ()),
        "vocab_size": int(text.args.vocab_size),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "targets_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[dwq-collect-dsv4] done in {meta['elapsed_sec']}s -> {out}", flush=True)


def _topk(logits, k):
    idx = mx.argpartition(-logits, k, axis=-1)[..., :k]
    vals = mx.take_along_axis(logits, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    return mx.take_along_axis(vals, order, axis=-1), mx.take_along_axis(idx, order, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--force-token-ids", default="1",
                    help="Comma-separated token ids forced into the retained "
                         "top-k at every position (default EOS=1). See "
                         "decision-critical support omission, arXiv 2607.07050.")
    ap.add_argument("--layers-at-once", type=int, default=1)
    ap.add_argument("--exclude-categories", nargs="*", default=["multimodal"])
    a = ap.parse_args()
    collect_targets(a.teacher, a.dataset, a.out, a.n_prompts, a.max_tokens,
                    a.topk, a.layers_at_once, a.exclude_categories,
                    tuple(int(x) for x in a.force_token_ids.split(',') if x.strip()))


if __name__ == "__main__":
    main()
