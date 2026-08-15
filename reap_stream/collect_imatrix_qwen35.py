"""Collect an oQe-compatible importance matrix for Qwen3.5/3.6 checkpoints.

Deliberately NOT the streaming collector. `collect_imatrix_streamed.py` drives
the model block-by-block because DeepSeek-V4-Flash is 155GB against 128GB of
RAM; that forced a hand-written forward pass, which is only safe because
DeepSeek's block signature is simple (h, mask, cache, input_ids).

Qwen3.5 is 55.6GB -- it loads outright. And its forward is much richer: two
different masks (SSM vs full-attention, `full_attention_interval: 4`), rope
position_embeddings, gdn_sink/target_verify plumbing, and per-layer
ArraysCache-vs-KVCache selection. Reimplementing that by hand to save memory we
don't need would be inventing a way to get it subtly wrong. So: install the
probes, call the model's own __call__, let it do its job.

Usage:
    PYTHONPATH=/Applications/oMLX.app/Contents/Resources .venv/bin/python \
        -m reap_stream.collect_imatrix_qwen35 \
        --model ~/Desktop/models/Qwen3.8-27B \
        --out artifacts/imatrix_qwen38.npz --num-samples 128 --seq-length 512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

OMLX = Path("/Applications/oMLX.app/Contents/Resources")


def _oq():
    if str(OMLX) not in sys.path:
        sys.path.insert(0, str(OMLX))
    import omlx.oq as oq
    return oq


def _prompts(dataset: str, n: int, tok, seq_length: int):
    """Prefer a local corpus; otherwise fall back to oQ's own calibration set.

    Note the DeepSeek agentic corpus is PRERENDERED with DeepSeek special
    tokens, which are meaningless to Qwen's tokenizer -- so a Qwen run should
    normally use oQ's bundled multi-category data rather than that file.
    """
    texts: list[str] = []
    p = Path(dataset) if dataset else None
    if p and p.exists():
        for line in p.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec.get("text") or rec.get("rendered") or ""
            if t.strip():
                texts.append(t)
            if len(texts) >= n:
                break
    if not texts:
        data = json.loads((OMLX / "omlx" / "oqe_calibration_data.json").read_text())
        cats = [c for c in ("tool_calling", "code", "reasoning", "chat", "en") if c in data]
        # round-robin the categories so a short sample stays balanced
        pools = [list(data[c]) for c in cats]
        i = 0
        while len(texts) < n and any(pools):
            pool = pools[i % len(pools)]
            if pool:
                texts.append(pool.pop(0))
            i += 1
        print(f"[imatrix] using oQ bundled calibration ({', '.join(cats)})", flush=True)
    return [tok.encode(t)[:seq_length] for t in texts[:n] if t.strip()]


def run(model_path: str, dataset: str, out_path: str,
        num_samples: int, seq_length: int) -> None:
    oq = _oq()
    t0 = time.time()

    model, processor = load(model_path, lazy=False)
    tok = getattr(processor, "tokenizer", processor)
    print(f"[imatrix] loaded, active={mx.get_active_memory()/1e9:.1f}GB "
          f"({time.time()-t0:.0f}s)", flush=True)

    prompts = [p for p in _prompts(dataset, num_samples, tok, seq_length) if len(p) > 8]
    print(f"[imatrix] {len(prompts)} samples, seq<={seq_length}", flush=True)

    collector = oq.OQImatrixCollector()
    n_probes = collector.install(model)
    print(f"[imatrix] installed {n_probes} probes "
          f"({dict(collector.capture_module_classes)})", flush=True)

    lm = getattr(model, "language_model", None) or model
    for i, toks in enumerate(prompts):
        ids = mx.array(toks)[None]
        mx.eval(lm(ids))
        if i % 16 == 0 or i == len(prompts) - 1:
            print(f"[imatrix] {i+1}/{len(prompts)} entries={len(collector.entries)} "
                  f"active={mx.get_active_memory()/1e9:.1f}GB ({time.time()-t0:.0f}s)",
                  flush=True)
        mx.clear_cache()

    entries = collector.entries
    if not entries:
        raise RuntimeError("no imatrix entries -- probes never fired")

    cfg = json.loads((Path(model_path).expanduser() / "config.json").read_text())
    meta = oq._source_imatrix_signature(
        Path(model_path).expanduser(), cfg,
        num_samples=len(prompts), seq_length=seq_length,
        calib_dataset=oq._OQE_CALIB_DATASET,
    )
    coverage = {n: int(np.count_nonzero(e.counts)) for n, e in entries.items()
                if getattr(e.counts, "size", 1) > 1}
    meta.update({
        "entry_count": len(entries),
        "collection": {"streamed": False, "source": dataset or "oq-bundled",
                       "processed_samples": len(prompts)},
        "expert_coverage": coverage,
        "requires_expert_counts": bool(coverage),
        "processed_samples": len(prompts),
    })
    out = Path(out_path).expanduser()
    oq._save_oqe_imatrix(out, entries, meta)
    dense = sum(1 for e in entries.values() if np.ndim(e.in_sum2) == 1)
    print(f"[imatrix] wrote {out} ({len(entries)} entries, {dense} dense) "
          f"in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--seq-length", type=int, default=512)
    a = ap.parse_args()
    run(a.model, a.dataset, a.out, a.num_samples, a.seq_length)


if __name__ == "__main__":
    main()
