"""Text-only ORPO LoRA training for DeepSeek-V4-Flash (mlx_lm-native).

mlx_lm.lora has no ORPO/preference loss at all (checked the installed 0.31.3:
tuner/losses.py has only kl_div_loss/js_div_loss, tuner/trainer.py has exactly
one loss, default_loss, plain SFT). The only real ORPO implementation in this
stack is mlx_vlm.trainer.orpo_trainer, and it's hard-wired to a VLM forward
call (get_logps does `model(shifted_input_ids, pixel_values,
shifted_attention_mask, **kwargs)`, pixel_values always positional).
DeepSeek-V4 has zero vision tensors and loads through the omlx-patched
mlx_lm TEXT path (mlx_lm.models.deepseek_v4.Model.__call__(self, inputs,
cache=None) -> raw mx.array, no .logits wrapper, no attention_mask
parameter at all -- verified against the real class before writing this).

So: reuse orpo_loss() verbatim from mlx_vlm (the odds-ratio math is pure
array arithmetic, model-agnostic) and write a text-only get_logps that calls
the model the way it actually expects. Same shape of fix scripts/
orpo_vlm_local.py already applied once for Step-3.7's own get_logps quirk,
just without the pixel_values plumbing to route around.

Loading needs omlx's deepseek_v4 patch applied first (mlx_lm doesn't know
the model_type exists otherwise) -- omlx itself isn't pip-installed in this
venv, but omlx.patches.deepseek_v4 imports cleanly standalone once its repo
root is on sys.path (verified: apply_deepseek_v4_patch() runs end to end,
including the omlx.cache.type_registry cache-handler step, which is serving-
only and irrelevant here but doesn't error either).

LoRA target keys are scoped to primary attention only (wq_a, wq_b, wkv,
wo_b -- 43 layers x 4 = 172 modules, verified via named_modules() before
writing this) not the full linear_to_lora_layers default of every nn.Linear
in scope, which would also wrap all 256 experts/layer. The behavior being
trained (closing-bracket count in multi-level nesting, see
docs/DEEPSEEK-V4-FINDINGS.md) is a state-tracking pattern, i.e. attention's
job, not a per-expert lookup -- and note wq_b/wkv/wgate also appear under
attn.indexer.* and attn.compressor.* (HISA's sparse-indexer side path);
mlx_lm's own key-matching is exact-string, not endswith, so the short keys
below only match the primary attn.* path, not those nested ones.

Usage:
    .venv/bin/python -m reap_stream.orpo_sft_deepseek_v4 \
        --model /Users/true/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-p37-native \
        --data data/lora_gemma4/orpo_paren_repair_pairs.jsonl \
        --adapter-dir artifacts/lora/deepseek_v4_paren_repair \
        --iters 30 --lr 1e-5 --max-seq-length 4096
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

OMLX_ROOT = "/Users/true/Documents/dsdsd/omlx"


def apply_patch():
    if OMLX_ROOT not in sys.path:
        sys.path.insert(0, OMLX_ROOT)
    import omlx.patches.deepseek_v4 as dsv4_patch
    dsv4_patch.apply_deepseek_v4_patch()
    return dsv4_patch


# Exact-string relative keys (see module docstring for why these and not
# the nested attn.indexer.* / attn.compressor.* siblings that share names).
LORA_KEYS = {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b"}


def get_logps_text(model, input_ids, mask, eps_tokens=1):
    """Text-only analog of mlx_vlm.trainer.orpo_trainer.get_logps: no
    pixel_values, no attention_mask forward arg (deepseek_v4's Model.__call__
    only takes inputs + cache), raw array return (no .logits wrapper)."""
    batch_size, seq_length = input_ids.shape
    shifted_input_ids = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    shifted_mask = mask[:, :-1]
    target_mask = mask[:, 1:]

    logits = model(shifted_input_ids).astype(mx.float32)

    if logits.shape[1] != targets.shape[1]:
        if logits.shape[1] < targets.shape[1]:
            pad = targets.shape[1] - logits.shape[1]
            logits = mx.pad(logits, ((0, 0), (0, pad), (0, 0)), constant_values=-100)
        else:
            logits = logits[:, -targets.shape[1]:, :]

    log_probs = -nn.losses.cross_entropy(logits, targets, reduction="none")
    mask_f = target_mask.astype(log_probs.dtype)
    token_counts = mx.maximum(mask_f.sum(-1), eps_tokens)
    logp_seq_avg = (log_probs * mask_f).sum(-1) / token_counts
    logits_mean = logits.sum() / mx.maximum(mask_f.sum(), 1)
    return logp_seq_avg, logits_mean, shifted_mask


def find_prompt_len(tok, messages, tools, max_len):
    """Same longest-common-prefix method reap_stream/lora_sft_step37.py
    already uses: render messages[:-1] with add_generation_prompt to find
    where the completion (final turn) starts."""
    try:
        full = tok.apply_chat_template(messages, tools=tools, tokenize=False)
        prompt = tok.apply_chat_template(
            messages[:-1], tools=tools, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        return None
    full_ids = tok.encode(full, add_special_tokens=False)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        p = 0
        for a, b in zip(full_ids, prompt_ids):
            if a != b:
                break
            p += 1
    if p >= len(full_ids):
        return None
    n_full = len(full_ids)
    if n_full > max_len:
        cut = n_full - max_len
        full_ids = full_ids[cut:]
        p = max(p - cut, 0)
    if p >= len(full_ids):
        return None
    completion_mask = [0] * p + [1] * (len(full_ids) - p)
    return full_ids, completion_mask


def tokenize_pair(tok, row, max_len):
    tools = None  # rows carry tools inline per-message context, not a separate field
    chosen = find_prompt_len(tok, row["chosen"], tools, max_len)
    rejected = find_prompt_len(tok, row["rejected"], tools, max_len)
    if chosen is None or rejected is None:
        return None
    return chosen, rejected


def pad_batch(items, pad_len):
    ids = np.zeros((len(items), pad_len), dtype=np.int32)
    mask = np.zeros((len(items), pad_len), dtype=np.float32)
    for i, (tok_ids, comp_mask) in enumerate(items):
        n = min(len(tok_ids), pad_len)
        ids[i, :n] = tok_ids[:n]
        mask[i, :n] = comp_mask[:n]
    return mx.array(ids), mx.array(mask)


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_probe_margins(model, tok, probe_rows, max_len):
    """Generalization probe: for each held-out row, tokenize the full
    chosen/rejected text and find where they diverge (longest common token
    prefix -- chosen/rejected differ only in the one corrected line, so this
    isolates exactly that span rather than diluting the signal across a long
    shared context, which get_logps_text's whole-completion masking would
    do for these specific rows -- they carry full real session context,
    sometimes 100k+ chars, per the real corpus). Returns list of
    (chosen_logp, rejected_logp, margin) per row."""
    def truncate_and_mask(ids_full, p):
        n = len(ids_full)
        if n > max_len:
            cut = n - max_len
            ids, p_trunc = ids_full[cut:], max(p - cut, 0)
        else:
            ids, p_trunc = ids_full, p
        mask = [0] * p_trunc + [1] * (len(ids) - p_trunc)
        return ids, mask

    results = []
    for row in probe_rows:
        c_full = tok.apply_chat_template(row["chosen"], tokenize=False)
        r_full = tok.apply_chat_template(row["rejected"], tokenize=False)
        c_ids_full = tok.encode(c_full, add_special_tokens=False)
        r_ids_full = tok.encode(r_full, add_special_tokens=False)
        p = 0
        for x, y in zip(c_ids_full, r_ids_full):
            if x != y:
                break
            p += 1
        if p >= len(c_ids_full) or p >= len(r_ids_full):
            continue  # identical -- nothing to compare

        c_ids, c_mask = truncate_and_mask(c_ids_full, p)
        r_ids, r_mask = truncate_and_mask(r_ids_full, p)
        c_arr, c_m = pad_batch([(c_ids, c_mask)], max_len)
        r_arr, r_m = pad_batch([(r_ids, r_mask)], max_len)
        c_logp, _, _ = get_logps_text(model, c_arr, c_m)
        r_logp, _, _ = get_logps_text(model, r_arr, r_m)
        c_val, r_val = float(c_logp.item()), float(r_logp.item())
        results.append((c_val, r_val, c_val - r_val))
    return results


def print_probe_report(label, results, rows):
    print(f"[orpo-dsv4] probe ({label}):", flush=True)
    for (c, r, m), row in zip(results, rows):
        tag = row["meta"].get("bad_line", "")[:60]
        print(f"    {tag!r:65s} chosen={c:.4f} rejected={r:.4f} margin={m:+.4f}", flush=True)
    if results:
        avg = sum(m for _, _, m in results) / len(results)
        print(f"    avg margin: {avg:+.4f} (positive = prefers correct bracket count)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="chosen/rejected jsonl (build_paren_repair_orpo.py output)")
    ap.add_argument("--probe-data", default=None, help="held-out generalization probe jsonl, same schema")
    ap.add_argument("--adapter-dir", default="artifacts/lora/deepseek_v4_orpo")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--steps-per-report", type=int, default=5)
    ap.add_argument("--steps-per-eval", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mx.random.seed(a.seed)
    np.random.seed(a.seed)

    print(f"[orpo-dsv4] applying omlx deepseek_v4 patch", flush=True)
    apply_patch()

    import mlx_lm
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
    from mlx_vlm.trainer.orpo_trainer import orpo_loss

    if mx.metal.is_available():
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

    print(f"[orpo-dsv4] loading {a.model}", flush=True)
    model, tok = mlx_lm.load(a.model)

    model.freeze()
    lora_cfg = {"rank": a.rank, "scale": a.scale, "dropout": a.dropout, "keys": LORA_KEYS}
    linear_to_lora_layers(model, -1, lora_cfg)
    print_trainable_parameters(model)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"[orpo-dsv4] trainable params: {n_train/1e6:.3f}M", flush=True)

    # ESSENTIAL, not optional -- reap_stream/lora_sft_step37.py hit 241GB
    # (>128GB RAM) without this at seq 8192 on a comparably deep model (45
    # vs this model's 43 layers). Distinct layer classes across the stack
    # (sliding/full/compressed attention variants) mean grad_checkpoint's
    # class-level patch needs to see one instance of each -- loop every
    # layer like step37 does, not just layers[0].
    n_ckpt = 0
    for layer in model.layers:
        if layer is not None:
            grad_checkpoint(layer)
            n_ckpt += 1
    print(f"[orpo-dsv4] gradient checkpointing on {n_ckpt} layers", flush=True)

    probe_rows = load_rows(a.probe_data) if a.probe_data else []
    if probe_rows:
        # Measured with the LoRA structure already attached (lora_b is
        # zero-init per mlx_lm.tuner.lora.LoRALinear convention, so this is
        # equivalent to the base model) -- avoids a second 96GB model load
        # just to get an "untrained" baseline.
        model.eval()
        baseline = compute_probe_margins(model, tok, probe_rows, a.max_seq_length)
        model.train()
        print_probe_report("BEFORE training", baseline, probe_rows)

    rows = load_rows(a.data)
    rng = np.random.RandomState(a.seed)
    idx = rng.permutation(len(rows))
    n_val = max(1, int(len(rows) * a.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    print(f"[orpo-dsv4] train {len(train_rows)} / val {len(val_rows)} pairs", flush=True)

    tokenized = {}
    def get_tok(i, rowset):
        key = (id(rowset), i)
        if key not in tokenized:
            tokenized[key] = tokenize_pair(tok, rowset[i], a.max_seq_length)
        return tokenized[key]

    pad_len = a.max_seq_length

    def batch_from(rowset, i):
        t = get_tok(i, rowset)
        if t is None:
            return None
        chosen, rejected = t
        c_ids, c_mask = pad_batch([chosen], pad_len)
        r_ids, r_mask = pad_batch([rejected], pad_len)
        return c_ids, c_mask, r_ids, r_mask

    def loss_fn_wrapper(c_ids, c_mask, r_ids, r_mask):
        chosen_logps, chosen_lm, chosen_final_mask = get_logps_text(model, c_ids, c_mask)
        rejected_logps, rejected_lm, rejected_final_mask = get_logps_text(model, r_ids, r_mask)
        losses, reward, num_tokens, metrics = orpo_loss(
            chosen_logps, chosen_lm, rejected_logps, rejected_lm,
            chosen_final_mask, rejected_final_mask, beta=a.beta,
        )
        return losses, num_tokens

    lg_fn = nn.value_and_grad(model, lambda c_ids, c_mask, r_ids, r_mask: loss_fn_wrapper(c_ids, c_mask, r_ids, r_mask)[0])
    opt = optim.AdamW(learning_rate=a.lr)

    ck = Path(a.adapter_dir)
    ck.mkdir(parents=True, exist_ok=True)

    def save():
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(ck / "adapters.safetensors"), weights)

    def run_eval():
        tot, k = 0.0, 0
        for i in range(len(val_rows)):
            b = batch_from(val_rows, i)
            if b is None:
                continue
            c_ids, c_mask, r_ids, r_mask = b
            tot += float(loss_fn_wrapper(c_ids, c_mask, r_ids, r_mask)[0].item())
            k += 1
            mx.clear_cache()
        return tot / max(k, 1)

    order = np.random.permutation(len(train_rows))
    ptr = 0
    t0 = time.time()
    ts = time.time()
    model.train()
    for it in range(1, a.iters + 1):
        b = None
        while b is None:
            if ptr >= len(order):
                order = np.random.permutation(len(train_rows))
                ptr = 0
            b = batch_from(train_rows, int(order[ptr]))
            ptr += 1
        c_ids, c_mask, r_ids, r_mask = b
        loss, grads = lg_fn(c_ids, c_mask, r_ids, r_mask)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state, loss)
        mx.clear_cache()

        if it <= 3 or it % a.steps_per_report == 0:
            print(f"[orpo-dsv4] iter {it}/{a.iters} loss={loss.item():.4f} "
                  f"({time.time()-ts:.1f}s)", flush=True)
            ts = time.time()

        if a.steps_per_eval and it % a.steps_per_eval == 0:
            model.eval()
            vl = run_eval()
            if probe_rows:
                pm = compute_probe_margins(model, tok, probe_rows, a.max_seq_length)
                avg_pm = sum(m for _, _, m in pm) / len(pm)
                per_row = " ".join(f"{m:+.3f}" for _, _, m in pm)
                print(f"[orpo-dsv4] iter {it} probe avg_margin={avg_pm:+.4f} ({per_row})", flush=True)
            model.train()
            print(f"[orpo-dsv4] iter {it} VAL loss={vl:.4f}", flush=True)

        if a.save_every and it % a.save_every == 0:
            save()
            print(f"[orpo-dsv4] saved adapter at iter {it} -> {ck}", flush=True)

    save()
    print(f"[orpo-dsv4] done in {time.time()-t0:.0f}s, final adapter -> {ck}", flush=True)

    if probe_rows:
        model.eval()
        after = compute_probe_margins(model, tok, probe_rows, a.max_seq_length)
        print_probe_report("AFTER training", after, probe_rows)
        for (bc, br, bm), (ac, ar, am), row in zip(baseline, after, probe_rows):
            tag = row["meta"].get("bad_line", "")[:60]
            print(f"    {tag!r:65s} margin {bm:+.4f} -> {am:+.4f} (delta {am-bm:+.4f})", flush=True)


if __name__ == "__main__":
    main()
