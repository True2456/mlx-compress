"""COCONUT-style curriculum training: retrofit continuous-thought latent
reasoning onto Gemma-4-12B via QLoRA, on our reconstructed full trajectories.

Mechanism (validated in smoke_continuous_thought.py -- gradient does flow
through this correctly): each intermediate assistant turn's nonempty free
text (its deliberation before a tool call, e.g. "I'll look at the wrapper,
the deployer, and the test gate.") is a compression candidate. At curriculum
stage k, the first k candidates (in trajectory order) are replaced: instead
of embedding their real tokens, the model's own final hidden state from the
end of the preceding context is fed back as the next input embedding, for
`continuous_thoughts_per_step` iterations, then teacher-forcing resumes with
whatever real tokens come next (the tool call, or the next segment). Loss is
cross-entropy on real-token positions only; continuous-thought positions
have no target and contribute no loss.

Deliberately NOT reusing collect_gemma4.py's layer-streaming machinery --
that's built for lazy windowed inference (freeing layers as it goes), which
is incompatible with needing the whole model resident for backprop.

Only nonempty assistant `content` spans are compression candidates -- most
turns in these trajectories are empty (just a tool call), and the final
assistant turn is never touched (it's the actual target, matching
mask_prompt's own convention of always preserving the true answer span).

Usage:
    .venv/bin/python -m reap_stream.lora_loop_gemma4 \
        --model ~/.lmstudio/models/mlx-community/gemma-4-12B-it-bf16 \
        --data data/loop_curriculum_data \
        --adapter-path adapters/gemma4-12b-loop \
        --stage 0 --iters 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import grad_checkpoint

# This cap is set by the machine's RAM ceiling, not by data coverage.
#
# Measured (12B bf16, rank-8 LoRA on all attn+mlp, gradient checkpointing on,
# 1 continuous thought), real phys_footprint_peak vs sequence length:
#     1024 tok ->  45 GB,  3.38s
#     2048 tok ->  66 GB,  6.91s
#     3072 tok ->  88 GB, 10.67s
# Both are LINEAR up to 3072: ~21 GB and ~3.5s per 1024 tokens. Then 4096
# breaks trend hard -- 74.84s against a ~14s extrapolation (5.3x), at 110 GB
# real peak on a 128 GB machine. Measured twice: 145.52s with prior swap
# pressure on the machine, 74.84s from a clean state, so treat single
# measurements near the ceiling with suspicion. Swap barely moved during the
# clean run, which points at macOS memory compression rather than paging.
#
# Choosing the cap is then arithmetic over the length distribution, because
# only the tail is expensive (median trajectory is ~1400-1700 tokens):
#     cap 3072 -> 434 rows, 131 fewer than 4096, blended ~7s/iter
#     cap 4096 -> 565 rows, but 23% of them sit above 3072 at ~75s each,
#                 blending out to ~23.5s/iter
# 3.4x the per-iteration cost for 30% more data is a bad trade.
#
# Data coverage lost here is real -- p50 trajectory length across the corpus is
# 8016 tokens, so this keeps only 434/2512 (17%). Buying that back means
# reducing activation memory (~21 GB per 1024 tokens is high for a 12B with
# gradient checkpointing on), not raising this number.
MAX_TRAJECTORY_TOKENS = 3072
# Measured eligible-segment distribution at the 8192 cap: mean 1.9, 56% of rows
# have >=2, only 6% have >=4. The curriculum cannot meaningfully go past ~stage 3
# on this data -- this cap is generous, not binding.
MAX_ELIGIBLE_SEGMENTS = 8

# Cross-entropy is computed over chunks of this many positions, each chunk's
# vocab projection wrapped in mx.checkpoint. Gemma-4's vocab is 262144, so a
# full-sequence logits tensor at 8192 tokens would be ~4GB in bf16 (and more
# once cross_entropy upcasts), retained for backward. Chunking bounds that to
# one chunk's worth at the cost of recomputing the projection in the backward
# pass.
LOGIT_CHUNK = 512


def _text_lm(model):
    return getattr(model, "language_model", None) or model


def _save_dtype(params):
    """LoRALinear initializes in float32, so trainable params save at 4 bytes
    each -- 2.1GB per checkpoint at rank 128. bf16 halves that at no meaningful
    cost: these are deltas applied to a bf16 base, and the optimizer's own
    state (which is what actually wants extra precision) is held in memory and
    never written here anyway.
    """
    return {k: v.astype(mx.bfloat16) for k, v in tree_flatten(params)}


def load_trajectories(path: str, tokenizer, max_tokens: int) -> list[dict]:
    """Render each trajectory once to find eligible (compressible) segment
    boundaries in token space, and drop anything too long to be practical."""
    rows = [json.loads(l) for l in open(path)]
    out = []
    for r in rows:
        messages = r["messages"]
        # Assistant turns carrying compressible free text, excluding the final
        # message (never compressed -- it's the real target). See segment_texts:
        # this counts reasoning_content as well as content, which raises mean
        # eligible segments per trajectory from 2.42 to 2.93 corpus-wide.
        eligible = [
            i for i, m in enumerate(messages[:-1])
            if m["role"] == "assistant" and segment_texts(m)
        ]
        if not eligible:
            continue
        eligible = eligible[:MAX_ELIGIBLE_SEGMENTS]

        try:
            full_text = tokenizer.apply_chat_template(
                messages, tools=r.get("tools"), tokenize=False
            )
        except Exception:
            continue
        n_tokens = len(tokenizer.encode(full_text))
        if n_tokens > max_tokens:
            continue

        out.append({"messages": messages, "tools": r.get("tools"),
                    "eligible": eligible, "n_tokens": n_tokens})
    return out


def segment_texts(msg) -> list[str]:
    """The compressible free text in an assistant turn, in render order.

    `reasoning_content` renders inside <|channel>thought ... <channel|> and is
    the closest thing in this corpus to COCONUT's actual target: deliberation
    emitted before an action. 1460 turns carry it with no `content` at all, so
    keying eligibility on `content` alone made them invisible to the curriculum.

    Deliberately EXCLUDES tool_calls: they are the action the rest of the
    trajectory depends on. Replacing a tool call with continuous thoughts
    leaves the following tool-response message with nothing that produced it.
    """
    out = []
    for field in ("reasoning_content", "content"):
        t = (msg.get(field) or "").strip()
        if t:
            out.append(t)
    return out


def _text_token_spans(tokenizer, messages, tools, seg_idx, full_text, offsets):
    """Token ranges [start, end) covering ONLY the free-text portions of
    message[seg_idx], never its tool calls.

    The previous implementation diffed rendered-prefix lengths, which yields
    the span of the WHOLE message -- and 41% of eligible segments also carry a
    tool call, so that silently compressed away the action too. This instead
    locates each text field's character range inside the message's own slice of
    the full rendering (the template is a strict character-level prefix per
    message, verified), then maps character offsets to token offsets using the
    fast tokenizer's offset mapping.
    """
    msg_start = len(tokenizer.apply_chat_template(messages[:seg_idx], tools=tools, tokenize=False))
    msg_end = len(tokenizer.apply_chat_template(messages[:seg_idx + 1], tools=tools, tokenize=False))
    window = full_text[msg_start:msg_end]

    spans = []
    search_from = 0
    for text in segment_texts(messages[seg_idx]):
        rel = window.find(text, search_from)
        if rel < 0:
            # Template may normalize whitespace; skip rather than guess at a
            # span, since a wrong span corrupts the example silently.
            continue
        c0, c1 = msg_start + rel, msg_start + rel + len(text)
        search_from = rel + len(text)

        tok_start = tok_end = None
        for ti, (a, b) in enumerate(offsets):
            if a == b:
                continue  # zero-width (special tokens) carry no character range
            if tok_start is None and b > c0:
                tok_start = ti
            if a < c1:
                tok_end = ti + 1
        if tok_start is not None and tok_end is not None and tok_end > tok_start:
            spans.append((tok_start, tok_end))
    return spans


def build_example(tokenizer, row: dict, stage: int, continuous_thoughts_per_step: int):
    """Returns (full_token_ids, replace_spans) where replace_spans is a list
    of (start, end) token ranges (in the ORIGINAL full rendering) to replace
    with continuous thoughts, for the first `stage` eligible segments."""
    messages, tools, eligible = row["messages"], row["tools"], row["eligible"]
    full_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    n_replace = min(stage, len(eligible))
    spans = []
    for seg_idx in eligible[:n_replace]:
        spans.extend(_text_token_spans(tokenizer, messages, tools, seg_idx, full_text, offsets))
    spans.sort()

    # A turn can contribute two spans (reasoning_content and content); they are
    # disjoint by construction, but assert it since overlapping spans would
    # silently corrupt the embedding sequence in forward_with_continuous_thoughts.
    for (s0, e0), (s1, _) in zip(spans, spans[1:]):
        assert e0 <= s1, f"overlapping compression spans: {(s0, e0)} vs {(s1, _)}"
    return full_ids, spans


def forward_with_continuous_thoughts(text_model, full_ids, spans, c_per_step, window_size):
    """Build the embedding sequence: real token embeddings everywhere except
    `spans`, which get `c_per_step` continuous-thought embeddings each,
    generated by the validated iterative hidden-state-feedback mechanism.

    Returns (logits, target_ids, loss_mask) aligned for next-token loss --
    loss_mask is 0 at continuous-thought positions (no target token exists)
    and at the position immediately before a replaced span (its "next token"
    target is meaningless once that span is gone).
    """
    embed_tokens = text_model.embed_tokens
    scale = getattr(text_model, "embed_scale", None)

    pieces_embeds = []   # list of mx.array chunks to concatenate
    pieces_targets = []  # parallel token-id chunks (or None for continuous-thought chunks)
    pos = 0
    for start, end in spans:
        if start > pos:
            real_ids = full_ids[pos:start]
            e = embed_tokens(mx.array(real_ids)[None])
            if scale is not None:
                e = e * scale
            pieces_embeds.append(e)
            pieces_targets.append(real_ids)
        pieces_embeds.append(("CT", c_per_step))  # placeholder, filled in during the run
        pieces_targets.append(None)
        pos = end
    if pos < len(full_ids):
        real_ids = full_ids[pos:]
        e = embed_tokens(mx.array(real_ids)[None])
        if scale is not None:
            e = e * scale
        pieces_embeds.append(e)
        pieces_targets.append(real_ids)

    def run_stack(h):
        # Masks depend only on sequence length and layer TYPE, of which there
        # are two -- so build them once per pass, not once per layer. The
        # sliding mask in particular is a materialized NxN array whenever
        # N > window_size (1024 here), so rebuilding it for each of the ~40
        # sliding layers was allocating the same multi-million-element mask
        # dozens of times per forward. This mirrors what Gemma4TextModel's own
        # _make_masks does (it dedupes by layer_type for exactly this reason).
        m_full = create_attention_mask(h, None)
        m_slide = create_attention_mask(h, None, window_size=window_size)
        hidden = h
        for layer in text_model.layers:
            m = m_slide if layer.layer_type == "sliding_attention" else m_full
            hidden, _, _ = layer(hidden, mask=m, cache=None)
        return text_model.norm(hidden)

    h = None
    all_target_ids = []
    loss_mask = []
    for piece, targets in zip(pieces_embeds, pieces_targets):
        if isinstance(piece, tuple) and piece[0] == "CT":
            c = piece[1]
            for _ in range(c):
                hidden = run_stack(h)
                next_embed = hidden[:, -1:, :]
                h = next_embed if h is None else mx.concatenate([h, next_embed], axis=1)
                all_target_ids.append(-1)  # no target at a continuous-thought position
                loss_mask.append(0)
                mx.clear_cache()  # each CT step is a full-stack pass; don't let freed buffers pile up mid-example
            # Conservative simplification: the last continuous-thought position's
            # logits could in principle predict the first real token that resumes
            # after it (real, available signal), but that's left out of the loss
            # here rather than risk getting the off-by-one wrong in a first pass.
            # Costs some supervision, not correctness.
        else:
            h = piece if h is None else mx.concatenate([h, piece], axis=1)
            all_target_ids.extend(targets)
            loss_mask.extend([1] * len(targets))

    hidden = run_stack(h)
    return hidden, all_target_ids, loss_mask


def compute_loss(model, text_model, full_ids, spans, prompt_len, c_per_step,
                 window_size, softcap=None):
    hidden, target_ids, loss_mask = forward_with_continuous_thoughts(
        text_model, full_ids, spans, c_per_step, window_size
    )
    # next-token shift, done on the hidden states so the vocab projection is
    # never materialized for the whole sequence at once.
    pred_hidden = hidden[:, :-1, :]
    seq_len = pred_hidden.shape[1]
    targets = mx.array([t if t != -1 else 0 for t in target_ids[1:]])
    mask = mx.array(loss_mask[1:], dtype=mx.float32)
    # also zero out positions before prompt_len (mask_prompt convention)
    if prompt_len > 0:
        prefix_mask = mx.arange(seq_len) >= (prompt_len - 1)
        mask = mask * prefix_mask.astype(mx.float32)

    def chunk_ce(h_chunk, t_chunk, m_chunk):
        logits = text_model.embed_tokens.as_linear(h_chunk)
        if softcap is not None:
            # Gemma-4 applies final_logit_softcapping (30.0) at inference via
            # LanguageModel.logits_from_hidden. Training against uncapped
            # logits would fit the adapter to a different output transform
            # than the one used at deploy time.
            logits = mx.tanh(logits / softcap) * softcap
        ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), t_chunk)
        return (ce * m_chunk).sum()

    ck = mx.checkpoint(chunk_ce)
    total = mx.zeros(())
    for s in range(0, seq_len, LOGIT_CHUNK):
        e = min(s + LOGIT_CHUNK, seq_len)
        total = total + ck(pred_hidden[:, s:e, :], targets[s:e], mask[s:e])
    return total / mx.maximum(mask.sum(), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/loop_curriculum_data")
    ap.add_argument("--adapter-path", default="adapters/gemma4-12b-loop")
    ap.add_argument("--stage", type=int, required=True, help="curriculum stage: number of leading eligible segments to compress")
    ap.add_argument("--continuous-thoughts-per-step", type=int, default=1)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--unfreeze-first-layers", type=int, default=0,
                     help="fully train the first N decoder layers instead of "
                          "LoRA-wrapping them (~1.8GB each). These are the layers "
                          "that must learn to accept fed-back hidden states as "
                          "input embeddings, which is where this mechanism's "
                          "distribution shift actually lands.")
    ap.add_argument("--lora-scale", type=float, default=2.0)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--max-trajectory-tokens", type=int, default=MAX_TRAJECTORY_TOKENS)
    ap.add_argument("--resume-adapter-file", default=None,
                     help="load previously-trained LoRA weights before continuing "
                          "-- needed to actually progress through curriculum stages "
                          "rather than starting each stage from a fresh, untrained adapter")
    a = ap.parse_args()

    print(f"[loop-train] loading {a.model} ...", flush=True)
    model, processor = load(a.model, lazy=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    text_model = _text_lm(model).model

    print(f"[loop-train] loading trajectories from {a.data} ...", flush=True)
    train_rows = load_trajectories(f"{a.data}/train.jsonl", tokenizer, a.max_trajectory_tokens)
    print(f"[loop-train] {len(train_rows)} usable trajectories after length filter", flush=True)
    assert train_rows, "no trajectories survived filtering"

    # Full fine-tuning is not an option at this scale: the 12B body is 10.76B
    # params, so weights + grads + Adam's two states in bf16 is 86.1 GB before
    # a single activation, against ~64 GB of activations at 3072 tokens on a
    # 128 GB machine. LoRA rank is the cheap capacity dial instead -- rank 128
    # is 523.8M trainable params (16x rank 8) for only ~3.1 GB of optimizer
    # state. Fully unfreezing individual layers costs ~1.8 GB each and is
    # reserved for the FIRST layers, where the mechanism's actual distribution
    # shift lands: those layers have to learn to accept the model's own hidden
    # states as input embeddings, which is a change to the input distribution
    # rather than to behaviour further up the stack.
    print(f"[loop-train] wrapping model in LoRA rank={a.lora_rank} "
          f"(fully unfreezing first {a.unfreeze_first_layers} layers)...", flush=True)
    model.freeze()
    full_ft_layers = set(range(a.unfreeze_first_layers))
    lora_modules = []
    for li, layer in enumerate(text_model.layers):
        if li in full_ft_layers:
            # Trained directly -- wrapping these in LoRA too would be redundant
            # (both the base weight and its low-rank delta trainable at once).
            layer.unfreeze()
            continue
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            base = getattr(layer.self_attn, name, None)
            if base is not None and not isinstance(base, LoRALinear):
                wrapped = LoRALinear.from_base(base, r=a.lora_rank, scale=a.lora_scale, dropout=0.0)
                setattr(layer.self_attn, name, wrapped)
                lora_modules.append(wrapped)
        for name in ["gate_proj", "up_proj", "down_proj"]:
            base = getattr(layer.mlp, name, None)
            if base is not None and not isinstance(base, LoRALinear):
                wrapped = LoRALinear.from_base(base, r=a.lora_rank, scale=a.lora_scale, dropout=0.0)
                setattr(layer.mlp, name, wrapped)
                lora_modules.append(wrapped)
    for m in lora_modules:
        m.unfreeze(recurse=False, keys=["lora_a", "lora_b"])
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"[loop-train] {len(lora_modules)} LoRA-wrapped layers, {n_trainable/1e6:.1f}M trainable params", flush=True)

    # grad_checkpoint patches type(layer).__call__, i.e. it is a CLASS-level
    # change that covers every layer instance at once -- which is why mlx_lm's
    # own trainer calls it exactly once on layers[0]. Calling it per layer (as
    # this did originally) nests 48 checkpoint wrappers around the same
    # function, so each backward re-runs the whole nested chain. Measured on a
    # 2048-token example: 115.67s per fwd+bwd nested vs 7.43s called once, at
    # an identical 61.0GB MLX peak -- a 15.6x slowdown that bought no memory
    # saving whatsoever.
    print("[loop-train] enabling gradient checkpointing (once, class-level) -- the "
          "continuous-thought loop does 2-3x+ the normal number of full-stack forward "
          "passes per example, so this isn't optional here.", flush=True)
    grad_checkpoint(text_model.layers[0])

    if a.resume_adapter_file:
        print(f"[loop-train] resuming from {a.resume_adapter_file} ...", flush=True)
        loaded = mx.load(a.resume_adapter_file)
        model.update(tree_unflatten(list(loaded.items())))
        mx.eval(model.parameters())

    window_size = getattr(text_model, "window_size", None)
    softcap = getattr(_text_lm(model), "final_logit_softcapping", None)
    print(f"[loop-train] window_size={window_size} final_logit_softcapping={softcap}", flush=True)
    optimizer = optim.Adam(learning_rate=a.learning_rate)

    def loss_fn(model, row):
        full_ids, spans = build_example(tokenizer, row, a.stage, a.continuous_thoughts_per_step)
        # prompt_len: token length of everything through the last user/tool
        # turn before the FIRST reasoning segment we might touch -- mirrors
        # mask_prompt's response-only convention, applied to the whole
        # trajectory rather than a single final turn.
        prompt_len = 0
        return compute_loss(model, text_model, full_ids, spans, prompt_len,
                            a.continuous_thoughts_per_step, window_size, softcap)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    out_dir = Path(a.adapter_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[loop-train] starting stage {a.stage}, {a.iters} iters...", flush=True)
    t_start = time.time()
    for it in range(1, a.iters + 1):
        t_iter = time.time()
        row = train_rows[(it - 1) % len(train_rows)]
        loss, grads = loss_and_grad(model, row)
        mx.eval(loss)
        optimizer.update(model, grads)
        # Only the LoRA deltas and the optimizer moments change per step; the
        # frozen 12B parameter tree is already materialized, so evaluating all
        # of model.parameters() every iteration just walks it for nothing.
        mx.eval(model.trainable_parameters(), optimizer.state)
        mx.clear_cache()
        dt = time.time() - t_iter
        if it % 5 == 0 or it == 1:
            # Per-iteration timing is the number that decides whether a given
            # curriculum length is affordable, so log it rather than making it
            # something you can only infer from wall clock afterwards.
            avg = (time.time() - t_start) / it
            eta_h = avg * (a.iters - it) / 3600
            print(f"[loop-train] iter {it}/{a.iters} stage={a.stage} "
                  f"loss={loss.item():.4f} ntok={row['n_tokens']} "
                  f"{dt:.1f}s (avg {avg:.1f}s, eta {eta_h:.1f}h)", flush=True)
        if it % a.save_every == 0:
            weights = _save_dtype(model.trainable_parameters())
            mx.save_safetensors(str(out_dir / f"stage{a.stage}_iter{it}_adapters.safetensors"), weights)
            print(f"[loop-train] saved -> {out_dir}/stage{a.stage}_iter{it}_adapters.safetensors", flush=True)

    weights = _save_dtype(model.trainable_parameters())
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), weights)
    print(f"[loop-train] done -> {out_dir}/adapters.safetensors", flush=True)


if __name__ == "__main__":
    main()
