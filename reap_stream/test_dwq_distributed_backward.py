"""Correctness test for cross-machine backward pass, before trusting it on
the real 43-layer DWQ training.

The one genuinely new thing distributed DWQ needs beyond the already-proven
AWQ distributed pipeline (awq_quantize_deepseek_v4_distributed.py, which only
sends activations FORWARD) is sending a GRADIENT backward across the machine
boundary: rank 0 (owns the later layers) computes d(loss)/d(hidden_boundary)
via mx.vjp treating the received activation as a differentiable input --
exactly like treating it as an extra function argument alongside the
trainable params -- and sends that gradient back to rank 1, which uses it as
the seed cotangent for its OWN mx.vjp to get gradients for its own layers'
params. This is standard reverse-mode chain-rule, MLX's mx.vjp is built for
exactly this ("given output cotangents, get input gradients"), but it's
untested in this codebase, so verify it against a same-process reference
before building the real thing on top of it.

Toy model (deterministic, no RNG, so both ranks trivially agree on the same
initial weights without needing synced random state):
    rank 1 (early "layer"):  h = relu(x @ W1 + b1)
    rank 0 (late "layer"):   y = h @ W2 + b2 ;  loss = mean((y - target)^2)

Each rank computes its own gradients via the distributed forward+backward,
then independently recomputes the FULL forward+backward locally (both ranks
have all the fixed values, so this needs no communication) as a ground-truth
reference, and asserts the distributed gradients match it exactly.

Usage:
    mlx.launch --hosts <rank0-ip>,<rank1-ip> --backend ring -- \
        .venv/bin/python -m reap_stream.test_dwq_distributed_backward
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _fixed_arrays():
    # Deterministic, not random -- both ranks must agree without communication.
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4) * 0.1
    W1 = mx.arange(16, dtype=mx.float32).reshape(4, 4) * 0.05 - 0.4
    b1 = mx.arange(4, dtype=mx.float32) * 0.01
    W2 = mx.arange(8, dtype=mx.float32).reshape(4, 2) * 0.03 - 0.1
    b2 = mx.arange(2, dtype=mx.float32) * 0.02
    target = mx.array([[0.5, -0.3], [0.1, 0.2]], dtype=mx.float32)
    return x, W1, b1, W2, b2, target


def _reference_grads(x, W1, b1, W2, b2, target):
    """Full forward+backward in one process, no distribution -- ground truth."""

    def full_loss(W1, b1, W2, b2):
        h = nn.relu(x @ W1 + b1)
        y = h @ W2 + b2
        return mx.mean((y - target) ** 2)

    loss, grads = mx.value_and_grad(full_loss, argnums=(0, 1, 2, 3))(W1, b1, W2, b2)
    return loss, grads  # (dW1, db1, dW2, db2)


def run() -> None:
    group = mx.distributed.init()
    if group is None:
        raise SystemExit("no distributed group -- run under mlx.launch")
    rank, size = group.rank(), group.size()
    if size != 2:
        raise SystemExit(f"this test expects exactly 2 ranks, got {size}")

    x, W1, b1, W2, b2, target = _fixed_arrays()
    ref_loss, (ref_dW1, ref_db1, ref_dW2, ref_db2) = _reference_grads(
        x, W1, b1, W2, b2, target
    )
    mx.eval(ref_loss, ref_dW1, ref_db1, ref_dW2, ref_db2)

    if rank == 1:
        # Early "layer": forward to produce h, send it, later receive dh,
        # backward through its own params using dh as the seed cotangent.
        def stage1_fwd(W1, b1):
            return nn.relu(x @ W1 + b1)

        # mx.vjp needs the cotangent up front in MLX's API; compute h first,
        # send it, then do the vjp once dh arrives.
        h = stage1_fwd(W1, b1)
        mx.eval(h)
        h_sent = mx.distributed.send(h, 0)
        mx.eval(h_sent)

        dh = mx.distributed.recv_like(h, 0)
        mx.eval(dh)

        _, (dW1, db1) = mx.vjp(stage1_fwd, (W1, b1), (dh,))
        mx.eval(dW1, db1)

        ok_w = bool(mx.allclose(dW1, ref_dW1, atol=1e-5).item())
        ok_b = bool(mx.allclose(db1, ref_db1, atol=1e-5).item())
        print(f"[rank1] dW1 matches reference: {ok_w}", flush=True)
        print(f"[rank1] db1 matches reference: {ok_b}", flush=True)
        print(f"[rank1] max|dW1-ref|={mx.max(mx.abs(dW1-ref_dW1)).item():.2e} "
              f"max|db1-ref|={mx.max(mx.abs(db1-ref_db1)).item():.2e}", flush=True)
        if not (ok_w and ok_b):
            raise SystemExit("[rank1] MISMATCH vs reference -- do not trust this pattern yet")

    else:
        # Late "layer": receive h (a differentiable INPUT, not a param),
        # forward, compute loss, backward w.r.t. (h, W2, b2) in one vjp call,
        # send dh back.
        h_shape = (x.shape[0], W1.shape[1])
        h_placeholder = mx.zeros(h_shape, dtype=mx.float32)
        h = mx.distributed.recv_like(h_placeholder, 1)
        mx.eval(h)

        def stage0_fwd(h, W2, b2):
            y = h @ W2 + b2
            return mx.mean((y - target) ** 2)

        (loss,), (dh, dW2, db2) = mx.vjp(stage0_fwd, (h, W2, b2), (mx.array(1.0),))
        mx.eval(loss, dh, dW2, db2)

        dh_sent = mx.distributed.send(dh, 1)
        mx.eval(dh_sent)

        ok_loss = bool(mx.allclose(loss, ref_loss, atol=1e-5).item())
        ok_w = bool(mx.allclose(dW2, ref_dW2, rtol=1e-3, atol=1e-6).item())
        ok_b = bool(mx.allclose(db2, ref_db2, rtol=1e-3, atol=1e-6).item())
        print(f"[rank0] loss matches reference: {ok_loss} "
              f"(distributed={loss.item():.6f} ref={ref_loss.item():.6f})", flush=True)
        print(f"[rank0] dW2 matches reference (rtol=1e-3): {ok_w}", flush=True)
        print(f"[rank0] db2 matches reference (rtol=1e-3): {ok_b}", flush=True)
        print(f"[rank0] dW2 dist={dW2.tolist()}", flush=True)
        print(f"[rank0] dW2 ref ={ref_dW2.tolist()}", flush=True)
        print(f"[rank0] db2 dist={db2.tolist()} ref={ref_db2.tolist()}", flush=True)
        print(f"[rank0] max|dW2-ref|={mx.max(mx.abs(dW2-ref_dW2)).item():.2e} "
              f"max|db2-ref|={mx.max(mx.abs(db2-ref_db2)).item():.2e}", flush=True)
        if not (ok_loss and ok_w and ok_b):
            raise SystemExit("[rank0] MISMATCH vs reference -- do not trust this pattern yet")

    print(f"[rank{rank}] PASS", flush=True)


if __name__ == "__main__":
    run()
