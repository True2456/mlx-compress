"""Teacher-scoring server for two-machine on-policy distillation. Runs on the
machine with enough unified memory to hold the teacher (e.g. the 128GB M5 Max
for a ~95GB teacher) -- loads it once, then answers scoring requests over a
Thunderbolt-bridge socket from distill_vlm_onpolicy_remote.py running the
student on a second machine. The teacher is never trained, so no optimizer
state / gradients live here -- just weights + one forward pass of activations.

Only sends back top-k logits (not the full vocab softmax) to keep each
response small (K logits+indices per position instead of ~150k), since
network payload isn't the bottleneck we're optimizing for here -- teacher
compute and student compute are.

Usage (on the teacher machine):
    .venv/bin/python scripts/distill_teacher_server.py \
        --model mlx-community/Step-3.7-Flash-bf16 --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import argparse
import logging
import socketserver

import numpy as np

import mlx.core as mx

from mlx_vlm.trainer.utils import Colors
from mlx_vlm.utils import load

from _kd_rpc import recv_msg, send_msg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_MODEL = None
_CONFIG = None


def _align_logits_with_targets(logits, targets):
    if logits.shape[1] < targets.shape[1]:
        pad_length = targets.shape[1] - logits.shape[1]
        pad_width = ((0, 0), (0, pad_length), (0, 0))
        return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
    if logits.shape[1] > targets.shape[1]:
        return logits[:, -targets.shape[1] :, :]
    return logits


def _forward_logits(model, config, shifted_input_ids, pixel_values, shifted_attention_mask):
    model_type = config.get("model_type") if isinstance(config, dict) else getattr(config, "model_type", None)
    attention_mask = None if model_type == "gemma4_unified" else shifted_attention_mask
    outputs = model(shifted_input_ids, pixel_values, attention_mask)
    return outputs.logits.astype(mx.float32)


def _score(request: dict) -> dict:
    input_ids = mx.array(request["input_ids"])
    pixel_values = mx.array(request["pixel_values"]) if request.get("pixel_values") is not None else None
    top_k = int(request["top_k"])

    shifted_input_ids = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    shifted_attention_mask = mx.ones_like(shifted_input_ids)

    logits = _forward_logits(_MODEL, _CONFIG, shifted_input_ids, pixel_values, shifted_attention_mask)
    logits = _align_logits_with_targets(logits, targets)

    vocab = logits.shape[-1]
    k = min(top_k, vocab)
    neg_logits = -logits
    idx_part = mx.argpartition(neg_logits, kth=k - 1, axis=-1)[..., :k]
    vals = mx.take_along_axis(logits, idx_part, axis=-1)
    mx.eval(vals, idx_part)

    return {
        "topk_values": np.array(vals).astype(np.float16),
        "topk_indices": np.array(idx_part).astype(np.uint32),
    }


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        peer = self.client_address[0]
        while True:
            try:
                request = recv_msg(self.request)
            except ConnectionError:
                logger.info(f"{Colors.OKBLUE}Connection closed by {peer}{Colors.ENDC}")
                return
            if request.get("op") == "ping":
                send_msg(self.request, {"ok": True})
                continue
            try:
                response = _score(request)
            except Exception as e:  # noqa: BLE001 -- report failure to client, don't crash the server
                logger.exception(f"{Colors.WARNING}Scoring request from {peer} failed{Colors.ENDC}")
                send_msg(self.request, {"error": str(e)})
                continue
            send_msg(self.request, response)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global _MODEL, _CONFIG

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()

    logger.info(f"{Colors.HEADER}Loading teacher from {a.model}{Colors.ENDC}")
    _MODEL, _ = load(a.model, processor_config={"trust_remote_code": True})
    _MODEL.eval()
    _CONFIG = _MODEL.config.__dict__ if hasattr(_MODEL.config, "__dict__") else _MODEL.config
    logger.info(f"{Colors.OKGREEN}Teacher loaded, serving on {a.host}:{a.port}{Colors.ENDC}")

    with Server((a.host, a.port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
