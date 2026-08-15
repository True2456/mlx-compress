"""On-policy VLM distillation, student side, for a two-machine split: this
process only ever loads the STUDENT (e.g. the 64GB M3 Max for a ~7GB student
+ full-fine-tune optimizer state), and gets teacher scoring over a socket
from distill_teacher_server.py running on a second machine (e.g. the 128GB
M5 Max holding a ~95GB teacher). Point --teacher-host at the server's
Thunderbolt-bridge IP.

Neither machine needs to hold both models, and neither model gets
reloaded between steps (the swap-per-batch alternative pays a full disk
reload of the 95GB teacher every cycle -- this avoids that entirely).

The teacher only sends back top-k logits per position (see
distill_teacher_server.py), so the KL loss here is computed over that
top-k support with the teacher's mass renormalized within it -- a standard
approximation for large-vocab distillation, not the full-vocab KL you'd get
running both models in-process (see distill_vlm_onpolicy.py for that
single-machine version).

Usage (on the student machine):
    .venv/bin/python scripts/distill_vlm_onpolicy_remote.py \
        -c artifacts/distill/onpolicy_remote.yaml --teacher-host 169.254.x.x

Example config:
    student_model: artifacts/students/step37_reap15
    data: artifacts/distill/prompts.jsonl
    output_dir: artifacts/distill/onpolicy_remote_ckpt
    iters: 2000
    learning_rate: 1e-6
    max_new_tokens: 256
    gen_temperature: 0.7
    kd_temperature: 1.0
    teacher_topk: 32
    teacher_port: 8765
    steps_per_report: 10
    save_every: 200
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
from itertools import cycle
from pathlib import Path

import numpy as np
import yaml

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer.datasets import NATIVE_PREPROCESS_MODELS
from mlx_vlm.trainer.utils import Colors
from mlx_vlm.utils import load, prepare_inputs, process_inputs_with_fallback, save_weights

from _kd_rpc import recv_msg, send_msg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeacherClient:
    def __init__(self, host: str, port: int, timeout: float = 120.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self._connect()

    def _connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        send_msg(self.sock, {"op": "ping"})
        reply = recv_msg(self.sock)
        if not reply.get("ok"):
            raise ConnectionError(f"teacher server at {self.host}:{self.port} did not respond to ping")

    def score(self, input_ids_np: np.ndarray, pixel_values_np, top_k: int) -> dict:
        request = {
            "input_ids": input_ids_np.astype(np.int32),
            "pixel_values": pixel_values_np.astype(np.float32) if pixel_values_np is not None else None,
            "top_k": top_k,
        }
        try:
            send_msg(self.sock, request)
            response = recv_msg(self.sock)
        except (ConnectionError, BrokenPipeError, OSError):
            logger.warning(f"{Colors.WARNING}Lost connection to teacher, reconnecting{Colors.ENDC}")
            self._connect()
            send_msg(self.sock, request)
            response = recv_msg(self.sock)
        if "error" in response:
            raise RuntimeError(f"teacher server error: {response['error']}")
        return response


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prompt_inputs(processor, config, prompt_text, images):
    model_type = config.get("model_type")
    inputs = None
    if model_type in NATIVE_PREPROCESS_MODELS:
        try:
            inputs = process_inputs_with_fallback(
                processor=processor,
                prompts=[prompt_text],
                images=images if images else None,
                audio=None,
                add_special_tokens=False,
            )
            if "images" in inputs and "pixel_values" not in inputs:
                inputs["pixel_values"] = inputs.pop("images")
        except Exception:
            inputs = None

    if inputs is None:
        inputs = prepare_inputs(
            processor=processor,
            images=images if images else None,
            audio=None,
            prompts=[prompt_text],
            image_token_index=config.get("image_token_index"),
            resize_shape=None,
        )
    return inputs


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--teacher-host", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    logger.info(f"{Colors.HEADER}Loading student from {cfg['student_model']}{Colors.ENDC}")
    model_s, processor = load(cfg["student_model"], processor_config={"trust_remote_code": True})
    student_config = model_s.config.__dict__ if hasattr(model_s.config, "__dict__") else model_s.config
    model_s.train()

    teacher_port = cfg.get("teacher_port", 8765)
    logger.info(f"{Colors.HEADER}Connecting to teacher at {a.teacher_host}:{teacher_port}{Colors.ENDC}")
    teacher = TeacherClient(a.teacher_host, teacher_port)

    rows = _load_jsonl(cfg["data"])
    logger.info(f"{Colors.OKBLUE}Loaded {len(rows)} prompts from {cfg['data']}{Colors.ENDC}")

    optimizer = optim.Adam(learning_rate=cfg["learning_rate"])
    T = cfg.get("kd_temperature", 1.0)
    top_k = cfg.get("teacher_topk", 32)
    max_new_tokens = cfg.get("max_new_tokens", 256)
    gen_temperature = cfg.get("gen_temperature", 0.7)
    steps_per_report = cfg.get("steps_per_report", 10)
    save_every = cfg.get("save_every", 200)
    iters = cfg["iters"]
    output_dir = Path(cfg["output_dir"])

    def loss_fn(model, shifted_input_ids, pixel_values, shifted_attention_mask, completion_mask, teacher_logp_topk, teacher_p_topk, teacher_topk_idx):
        s_logits = _forward_logits(model, student_config, shifted_input_ids, pixel_values, shifted_attention_mask)
        student_logp_full = nn.log_softmax(s_logits / T, axis=-1)
        student_logp_topk = mx.take_along_axis(student_logp_full, teacher_topk_idx, axis=-1)
        kl = (teacher_p_topk * (teacher_logp_topk - student_logp_topk)).sum(-1)
        mask_f = completion_mask.astype(kl.dtype)
        denom = mx.maximum(mask_f.sum(), 1)
        return (kl * mask_f).sum() / denom

    running_loss = 0.0
    reported = 0

    for it, row in zip(range(1, iters + 1), cycle(rows)):
        conversation = row["messages"]
        image_paths = row.get("images") or []
        num_images = len(image_paths)

        prompt_text = apply_chat_template(
            processor, student_config, conversation, num_images=num_images, add_generation_prompt=True
        )

        prompt_inputs = _prompt_inputs(processor, student_config, prompt_text, image_paths)
        prompt_ids = np.array(prompt_inputs["input_ids"]).reshape(1, -1)
        pixel_values_np = prompt_inputs.get("pixel_values")
        prompt_len = prompt_ids.shape[1]

        gen_token_ids = []
        for response in stream_generate(
            model_s, processor, prompt_text,
            image=image_paths or None,
            max_tokens=max_new_tokens,
            temperature=gen_temperature,
        ):
            if response.token is None:
                continue
            gen_token_ids.append(response.token)
            if response.finish_reason is not None:
                break

        if not gen_token_ids:
            logger.warning(f"{Colors.WARNING}Empty rollout at step {it}, skipping{Colors.ENDC}")
            continue

        full_ids_np = np.concatenate(
            [prompt_ids, np.array(gen_token_ids, dtype=np.int64)[None, :]], axis=1
        )
        pixel_values_arr = None
        if pixel_values_np is not None:
            pixel_values_arr = np.array(pixel_values_np)

        teacher_response = teacher.score(full_ids_np, pixel_values_arr, top_k)
        teacher_topk_values = mx.array(teacher_response["topk_values"].astype(np.float32))
        teacher_topk_idx = mx.array(teacher_response["topk_indices"].astype(np.int32))
        teacher_logp_topk = mx.stop_gradient(nn.log_softmax(teacher_topk_values / T, axis=-1))
        teacher_p_topk = mx.stop_gradient(mx.exp(teacher_logp_topk))

        full_ids = mx.array(full_ids_np)
        pixel_values = mx.array(pixel_values_arr) if pixel_values_arr is not None else None
        shifted_input_ids = full_ids[:, :-1]
        shifted_attention_mask = mx.ones_like(shifted_input_ids)

        completion_start = prompt_len - 1
        steps = mx.arange(shifted_input_ids.shape[1])[None, :]
        completion_mask = steps >= completion_start

        loss, grads = nn.value_and_grad(model_s, loss_fn)(
            model_s, shifted_input_ids, pixel_values, shifted_attention_mask,
            completion_mask, teacher_logp_topk, teacher_p_topk, teacher_topk_idx,
        )
        optimizer.update(model_s, grads)
        mx.eval(model_s.parameters(), optimizer.state)

        running_loss += loss.item()
        reported += 1

        if it % steps_per_report == 0:
            logger.info(
                f"{Colors.OKGREEN}step {it}/{iters}  kd_loss {running_loss / reported:.4f}  "
                f"completion_tokens {len(gen_token_ids)}{Colors.ENDC}"
            )
            running_loss = 0.0
            reported = 0

        if it % save_every == 0 or it == iters:
            ckpt_dir = output_dir / f"step_{it}"
            logger.info(f"{Colors.UNDERLINE}Saving checkpoint to {ckpt_dir}{Colors.ENDC}")
            save_weights(ckpt_dir, model_s)


if __name__ == "__main__":
    main()
