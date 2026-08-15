"""Quantize a carried Qwen3.5 MTP head to match the backbone's scheme.

Why: the MTP head carried out of the bf16 original stays unquantized while the
backbone is QuantizedLinear throughout (measured: 8 Linear vs 496
QuantizedLinear). oMLX's speculative verify path is built around quantized
modules (`_target_verify_quantized_argmax`, the `qwen35_verify_qmm` patch), and
every draft step re-reads 0.849GB of bf16 weights, so a bf16 drafter is a
plausible cause of MTP *reducing* throughput.

Safe to quantize aggressively: MTP drafts are verified by the target model via
rejection sampling, so drafter error costs accept-rate (speed), never output
correctness -- the same argument that justified 2/3-bit DSpark on DeepSeek-V4.

config.json keys MUST be `language_model.mtp.*`, not `mtp.*`. oMLX's sanitize
remaps the weights to that prefix, and the load-time class_predicate looks up
the remapped name; a `mtp.*` key silently misses, the module inits at the
default 4-bit, and loading a 6-bit packed weight into it is a shape error.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import mlx.core as mx

# Mirror the backbone recipe; `fc` is MTP-specific (concat[emb,hidden] entry
# projection) and gets extra bits since every draft flows through it.
MTP_RECIPE = {
    "self_attn.q_proj": (8, 64),
    "self_attn.k_proj": (8, 64),
    "self_attn.v_proj": (8, 64),
    "self_attn.o_proj": (4, 64),
    "mlp.gate_proj": (4, 128),
    "mlp.up_proj": (4, 128),
    "mlp.down_proj": (4, 128),
    "fc": (6, 128),
}


def _spec(name: str):
    best = None
    for k, v in MTP_RECIPE.items():
        if name.endswith(k + ".weight") and (best is None or len(k) > len(best[0])):
            best = (k, v)
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    M = Path(a.model).expanduser()

    shard = None
    tensors = {}
    for f in sorted(M.glob("*.safetensors")):
        w = mx.load(str(f))
        if any(k.startswith("mtp.") for k in w):
            shard = f
            tensors = w
            break
    if shard is None:
        raise SystemExit("no shard containing mtp.* found")

    out, qcfg, n = {}, {}, 0
    for k, v in tensors.items():
        spec = _spec(k) if k.startswith("mtp.") else None
        if spec is None or v.ndim != 2:
            out[k] = v
            continue
        bits, gs = spec
        if v.shape[-1] % gs:
            out[k] = v
            print(f"  skip {k} (in_dim {v.shape[-1]} % {gs})")
            continue
        wq, sc, bi = mx.quantize(v, bits=bits, group_size=gs)
        base = k[: -len(".weight")]
        out[f"{base}.weight"] = wq
        out[f"{base}.scales"] = sc
        out[f"{base}.biases"] = bi
        qcfg["language_model." + base] = {"bits": bits, "group_size": gs}
        n += 1
        print(f"  {bits}b/gs{gs:<3d} {k}")

    # mx.load returns arrays still backed by the source file; writing to that
    # same path truncates it before the lazy reads happen and every untouched
    # tensor comes back as zeros (measured: all 7 MTP norms -> 0.0000). Force
    # evaluation first, then write via a temp file and rename.
    mx.eval(list(out.values()))
    tmp = shard.with_suffix(".tmp.safetensors")
    mx.save_safetensors(str(tmp), out, metadata={"format": "mlx"})
    tmp.replace(shard)

    idx_p = M / "model.safetensors.index.json"
    idx = json.loads(idx_p.read_text())
    idx["weight_map"] = {k: v for k, v in idx["weight_map"].items()
                         if not k.startswith("mtp.")}
    for k in out:
        if k.startswith("mtp."):
            idx["weight_map"][k] = shard.name
    idx_p.write_text(json.dumps(idx, indent=2))

    cfg_p = M / "config.json"
    cfg = json.loads(cfg_p.read_text())
    cfg.setdefault("quantization", {}).update(qcfg)
    cfg_p.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    print(f"[mtp-q] quantized {n} modules; config keys written under "
          f"`language_model.mtp.*` ({len(qcfg)} entries)")


if __name__ == "__main__":
    main()
