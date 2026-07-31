"""Quant-damage tomography: locate WHERE 4-bit damage lives, by depth.

Builds sequential variants of the vblend student, each with one component
bumped to higher precision, runs the 500-prompt text PPL on each, then deletes
the model (disk holds only one variant at a time; builds are ~2-min artifacts).

Variants:
  w0..w5  : one 7-layer window of MoE experts (switch_mlp) at 6-bit
            windows: [3-9] [10-16] [17-23] [24-30] [31-37] [38-44]
  shared8 : share_expert + dense layers 0-2 MLPs + routers at 8-bit
            (runs on every token; deployable via stock per-module quant config)

Compare each variant's per-category NLL against artifacts/ppl-p15-vblend-500.json:
the drop vs vblend maps which depths/components hold the agentic +0.072 damage.

Usage:
    .venv/bin/python scripts/tomography_sweep.py [--only w2 shared8] [--keep]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_vlm.utils import (
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    skip_multimodal_module,
)
from mlx_vlm.quant_utils import quantize_model

PLAN = "artifacts/plans/plan_p15_blend03.json"
MODEL = "models/Step-3.7-Flash"
WINDOWS = [(3, 9), (10, 16), (17, 23), (24, 30), (31, 37), (38, 44)]


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _slice_first_dim(module, keep_idx, names=("weight", "scales", "biases", "bias")):
    for name in names:
        if name in module and module[name] is not None:
            module[name] = module[name][keep_idx]


def build_variant(out_dir: str, bits_for_path):
    """Same REAP-apply as build_student.py; bits_for_path(path) -> None (default
    4-bit) or dict of quant params for the bumped component."""
    plan = json.loads(Path(PLAN).read_text())
    new_n = len(next(iter(plan["layers"].values()))["keep"])
    src = get_model_path(MODEL)
    model, config, processor = fetch_from_hub(src, lazy=True, trust_remote_code=True)
    text = _text_model(model)
    for layer_key, layer_plan in plan["layers"].items():
        i = int(layer_key)
        keep = mx.array(layer_plan["keep"])
        moe = text.layers[i].mlp
        _slice_first_dim(moe.gate.gate, keep)
        moe.gate.router_bias = moe.gate.router_bias[keep]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            _slice_first_dim(getattr(moe.switch_mlp, proj), keep)
    for cfg in (config, config.get("text_config")):
        if isinstance(cfg, dict) and "moe_num_experts" in cfg:
            cfg["moe_num_experts"] = new_n
    for obj in (getattr(model, "config", None), getattr(text, "args", None),
                getattr(getattr(model, "language_model", None), "args", None)):
        if obj is not None and hasattr(obj, "moe_num_experts"):
            obj.moe_num_experts = new_n
        tc = getattr(obj, "text_config", None)
        if tc is not None and hasattr(tc, "moe_num_experts"):
            tc.moe_num_experts = new_n

    def predicate(path, module):
        if skip_multimodal_module(path):
            return False
        return bits_for_path(path) or True

    config.setdefault("vision_config", {})
    target, config = quantize_model(model, config, 64, 4, mode="affine",
                                    quant_predicate=predicate)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    save_weights(out, target, donate_weights=True)
    import glob
    for pattern in ("*.py", "*.json"):
        for f in glob.glob(str(src / pattern)):
            if Path(f).name == "model.safetensors.index.json":
                continue
            shutil.copy(f, out)
    for item in src.iterdir():
        if item.is_dir():
            dest = out / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
    # CORRECTED 2026-07-26 (see docs/TOKENIZER-INVESTIGATION.md's correction
    # section): save_pretrained() persists tokenizer_class="LlamaTokenizerFast",
    # a known name that makes AutoTokenizer (what mlx_lm/LM Studio calls)
    # apply Llama's own SentencePiece/Metaspace conversion instead of this
    # model's real pretokenizer. fix_tokenizer_class() removes that
    # declaration and verifies live.
    processor.save_pretrained(out)
    sys.path.insert(0, str(Path(__file__).parent))
    from fix_tokenizer_class import fix_tokenizer_class
    fix_tokenizer_class(out)
    save_config(config, config_path=out / "config.json")




def _layer_of(path: str) -> int | None:
    parts = path.split(".")
    for j, p in enumerate(parts):
        if p == "layers" and j + 1 < len(parts) and parts[j + 1].isdigit():
            return int(parts[j + 1])
    return None


def make_bits_fn(variant: str):
    if variant.startswith("w"):
        lo, hi = WINDOWS[int(variant[1:])]
        def fn(path):
            li = _layer_of(path)
            if li is not None and lo <= li <= hi and ".switch_mlp." in path:
                return {"group_size": 64, "bits": 6, "mode": "affine"}
            return None
        return fn
    if variant == "shared8":
        def fn(path):
            li = _layer_of(path)
            if ".share_expert." in path or ".gate.gate" in path:
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            if li is not None and li <= 2 and ".mlp." in path:
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return None
        return fn
    if variant == "head8":
        # The one member of the always-on weight class the sweep never covered.
        # SHARED8-RESULT.md's thesis is that components firing on every token
        # carry the quantization damage, because they get no implicit smoothing
        # from top-k partial activation. lm_head/embed_tokens are the extreme
        # case -- every token, and every vocabulary row participates in every
        # argmax -- but the build predicate returns bare True for them, so they
        # sit at 4-bit by default and were invisible to every earlier variant.
        # Costs +0.53 GB (1.06B params, 4.5 -> 8.5 bpw).
        def fn(path):
            if "lm_head" in path or "embed_tokens" in path:
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return None
        return fn
    if variant == "shared8-head8":
        base = make_bits_fn("shared8")
        def fn(path):
            hit = base(path)
            if hit is not None:
                return hit
            if "lm_head" in path or "embed_tokens" in path:
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return None
        return fn
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--keep", action="store_true", help="don't delete variant models")
    ap.add_argument("--build-one", default=None, help="internal: build this variant and exit")
    a = ap.parse_args()

    if a.build_one:
        build_variant(f"models/Step-3.7-tomo-{a.build_one}", make_bits_fn(a.build_one))
        return

    variants = a.only or [f"w{i}" for i in range(len(WINDOWS))] + ["shared8"]

    for v in variants:
        out_model = f"models/Step-3.7-tomo-{v}"
        out_json = f"artifacts/ppl-tomo-{v}-500.json"
        if Path(out_json).exists():
            print(f"[tomo] {v}: {out_json} exists, skipping", flush=True)
            continue
        print(f"[tomo] === {v}: building {out_model}", flush=True)
        # Build in a subprocess so MLX's hoarded buffers die with it before the
        # 97GB eval starts -- running both at once fills swap and risks the
        # jetsam kill / hard-reboot failure modes documented in FINDINGS.md 6.
        if not Path(out_model, "config.json").exists():
            r = subprocess.run([sys.executable, __file__, "--build-one", v])
            if r.returncode != 0:
                print(f"[tomo] {v}: BUILD FAILED (rc={r.returncode})", flush=True)
                break
        print(f"[tomo] === {v}: evaluating", flush=True)
        r = subprocess.run([
            sys.executable, "-m", "reap_stream.eval_ppl_streamed",
            "--model", out_model, "--out", out_json, "--n-prompts", "500",
        ])
        if r.returncode != 0:
            print(f"[tomo] {v}: EVAL FAILED (rc={r.returncode}), keeping model", flush=True)
            break
        if not a.keep:
            shutil.rmtree(out_model)
            print(f"[tomo] {v}: model deleted", flush=True)
        import gc
        gc.collect()
        mx.clear_cache()
    print("[tomo] sweep done", flush=True)


if __name__ == "__main__":
    main()
