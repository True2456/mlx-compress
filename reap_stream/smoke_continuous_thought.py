"""Isolated smoke test for the COCONUT mechanism: does gradient flow through
a chain of (run model -> take last hidden state -> feed back as next input
embedding -> run again) survive MLX's autodiff, all the way back to trainable
LoRA parameters?

Deliberately tiny and disposable -- this validates the single highest-risk
unknown before any curriculum/data-pipeline work is built around it. If this
doesn't work, nothing downstream matters.

Usage:
    .venv/bin/python -m reap_stream.smoke_continuous_thought \
        --model ~/.lmstudio/models/mlx-community/gemma-4-12B-it-bf16
"""
from __future__ import annotations

import argparse

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask


def _text_lm(model):
    return getattr(model, "language_model", None) or model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-continuous-thoughts", type=int, default=3)
    a = ap.parse_args()

    print("[smoke] loading model...", flush=True)
    model, processor = load(a.model, lazy=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    lm = _text_lm(model)
    text_model = lm.model  # Gemma4TextModel: has embed_tokens, layers, norm

    # Wrap a couple of layers in LoRA so there's something trainable to check
    # gradient flow into -- mirrors the real training setup, not the full
    # linear_to_lora_layers machinery, just enough to prove the point.
    from mlx_lm.tuner.lora import LoRALinear
    target = text_model.layers[-1].self_attn.q_proj
    lora_layer = LoRALinear.from_base(target, r=4, scale=2.0, dropout=0.0)
    text_model.layers[-1].self_attn.q_proj = lora_layer

    # Freeze everything, then unfreeze only the LoRA delta -- otherwise
    # trainable_parameters() covers the whole 12B model, which is both
    # wildly expensive for a smoke test and not the real training config.
    model.freeze()
    lora_layer.unfreeze(recurse=False, keys=["lora_a", "lora_b"])

    trainable = tree_flatten(model.trainable_parameters())
    print(f"[smoke] trainable params: {[n for n, _ in trainable]}", flush=True)
    assert trainable, "no trainable parameters found -- LoRA wrap failed"

    prompt = "The bug was caused by"
    ids = mx.array(tokenizer.encode(prompt))[None]

    def loss_fn(model):
        h = text_model.embed_tokens(ids)
        scale = getattr(text_model, "embed_scale", None)
        if scale is not None:
            h = h * scale

        # Run the real prompt tokens once to get a starting hidden state.
        mask = create_attention_mask(h, None)
        hidden = h
        for layer in text_model.layers:
            hidden, _, _ = layer(hidden, mask=mask, cache=None)
        hidden = text_model.norm(hidden)

        # Now the actual mechanism: feed the last hidden state back in as
        # the next input embedding, run the full stack again, repeat.
        for step in range(a.num_continuous_thoughts):
            next_embed = hidden[:, -1:, :]  # (batch, 1, hidden) continuous thought
            h = mx.concatenate([h, next_embed], axis=1)
            mask = create_attention_mask(h, None)
            hidden = h
            for layer in text_model.layers:
                hidden, _, _ = layer(hidden, mask=mask, cache=None)
            hidden = text_model.norm(hidden)

        # Dummy loss: just needs to be a scalar function of the final hidden
        # state, enough to check gradients propagate back through the whole
        # iterative chain to the LoRA weights.
        return (hidden.astype(mx.float32) ** 2).mean()

    print(f"[smoke] running {a.num_continuous_thoughts} continuous-thought steps + backward...", flush=True)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model)
    mx.eval(loss, grads)

    print(f"[smoke] loss = {loss.item():.6f}", flush=True)

    flat_grads = tree_flatten(grads)
    nonzero = 0
    total = 0
    for name, g in flat_grads:
        if g is None:
            continue
        total += 1
        gnorm = float(mx.sqrt((g.astype(mx.float32) ** 2).sum()).item())
        if gnorm > 0:
            nonzero += 1
        print(f"  grad[{name}] norm = {gnorm:.6f}", flush=True)

    print(f"\n[smoke] {nonzero}/{total} trainable tensors received nonzero gradient", flush=True)
    if nonzero == total and total > 0:
        print("[smoke] PASS -- gradient flows through the continuous-thought chain "
              "all the way back to LoRA parameters.")
    else:
        print("[smoke] FAIL -- gradient did not reach all trainable parameters. "
              "The mechanism as implemented does not work as-is.")


if __name__ == "__main__":
    main()
