"""Fuse a mlx_vlm LoRA adapter into its base model and save a standalone checkpoint.

mlx_vlm doesn't ship a `mlx_vlm.fuse` CLI (unlike mlx_lm). This loads the base
model with the adapter applied, fuses every LoRA-wrapped layer's delta into
the frozen weights via update_modules (mirrors mlx_lm/fuse.py's own
fuse-and-save logic, just loaded through mlx_vlm so vision_tower/embed_vision
survive untouched), then saves the merged weights/config and copies over the
unchanged tokenizer/processor files so the result is a complete,
directly-loadable checkpoint directory.

Usage:
    .venv/bin/python scripts/fuse_vlm_lora.py \
        --model ~/.lmstudio/models/mlx-community/gemma-4-12B-it-qat-4bit \
        --adapter-path adapters/gemma4-12b-agentic \
        --save-path models/gemma-4-12b-it-qat-4bit-frontierdistill-fused
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlx_vlm
import mlx_vlm.utils as u
from mlx.utils import tree_unflatten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--save-path", required=True)
    ap.add_argument(
        "--mlx-lm-style-keys", action="store_true",
        help="Adapter was trained via mlx_lm (not mlx_vlm) and its adapter_config.json "
             "'keys' are per-layer-relative names (e.g. 'self_attn.q_proj'). mlx_vlm's "
             "own apply_lora_layers/_apply_lora_layers misreads 'keys' as absolute paths "
             "from the model root and crashes with AttributeError -- verified against the "
             "real traceback, not assumed. adapter_utils.load_adapters + linear_to_lora_layers "
             "handles the relative-per-layer convention correctly, so use that path instead, "
             "targeted at model.language_model (where .layers actually lives on this "
             "gemma4_unified wrapper).",
    )
    a = ap.parse_args()

    model_path = str(Path(a.model).expanduser())

    print(f"[fuse] loading {model_path}", flush=True)
    if a.mlx_lm_style_keys:
        from mlx_vlm.trainer.adapter_utils import load_adapters
        model, processor = mlx_vlm.load(model_path, processor_config={"trust_remote_code": True})
        print(f"[fuse] applying mlx_lm-style adapter {a.adapter_path} to model.language_model", flush=True)
        load_adapters(model.language_model, a.adapter_path)
    else:
        model, processor = mlx_vlm.load(
            model_path, adapter_path=a.adapter_path, processor_config={"trust_remote_code": True}
        )

    fused_linears = [
        (n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")
    ]
    print(f"[fuse] fusing {len(fused_linears)} adapted layers", flush=True)
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    save_path = Path(a.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"[fuse] saving merged weights -> {save_path}", flush=True)
    u.save_weights(save_path, model)

    config = u.load_config(model_path)
    u.save_config(config, save_path / "config.json")

    src = Path(model_path)
    for f in src.iterdir():
        if f.suffix == ".safetensors" or f.name in ("model.safetensors.index.json", "config.json"):
            continue
        if f.is_file():
            shutil.copy2(f, save_path / f.name)
            print(f"[fuse] copied {f.name}", flush=True)

    print(f"[fuse] done -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
