"""Check whether mx.distributed.send/recv_like/all_gather -- as used inside
DeepseekV4Model.__call__'s own built-in pipeline path -- are differentiable.
If so, distributed DWQ training can just reuse the single-machine script's
nn.value_and_grad(lm, loss_fn) directly on a pipeline-sharded model, with no
manual vjp-at-the-boundary plumbing (test_dwq_distributed_backward.py's
approach) needed at all. If not, that manual approach is required.

Toy model exercising the SAME primitives DeepseekV4Model.__call__ uses at its
pipeline boundary (recv_like -> local compute -> send -> all_gather), so a
pass/fail here directly answers the question for the real model.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _fixed_arrays():
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4) * 0.1
    W1 = mx.arange(16, dtype=mx.float32).reshape(4, 4) * 0.05 - 0.4
    b1 = mx.arange(4, dtype=mx.float32) * 0.01
    W2 = mx.arange(8, dtype=mx.float32).reshape(4, 2) * 0.03 - 0.1
    b2 = mx.arange(2, dtype=mx.float32) * 0.02
    target = mx.array([[0.5, -0.3], [0.1, 0.2]], dtype=mx.float32)
    return x, W1, b1, W2, b2, target


def _reference_grads(x, W1, b1, W2, b2, target):
    def full_loss(W1, b1, W2, b2):
        h = nn.relu(x @ W1 + b1)
        y = h @ W2 + b2
        return mx.mean((y - target) ** 2)

    return mx.value_and_grad(full_loss, argnums=(0, 1, 2, 3))(W1, b1, W2, b2)


def run() -> None:
    group = mx.distributed.init()
    if group is None:
        raise SystemExit("no distributed group -- run under mlx.launch")
    rank, size = group.rank(), group.size()
    if size != 2:
        raise SystemExit(f"expected 2 ranks, got {size}")

    x, W1, b1, W2, b2, target = _fixed_arrays()
    ref_loss, (ref_dW1, ref_db1, ref_dW2, ref_db2) = _reference_grads(
        x, W1, b1, W2, b2, target
    )
    mx.eval(ref_loss, ref_dW1, ref_db1, ref_dW2, ref_db2)

    # Every rank builds BOTH sets of params (matching how a real sharded
    # model has every rank hold the full set of nn.Module trainable
    # parameters, just with some layers replaced by None/skipped) -- but
    # nn.value_and_grad only asks for gradients w.r.t. what's passed in, so
    # each rank's own params list stands in for "this rank's local layers".
    if rank == 1:
        params = {"W1": W1, "b1": b1}

        def loss_fn(params):
            h = nn.relu(x @ params["W1"] + params["b1"])
            h_sent = mx.distributed.send(h, 0)
            # Mirrors DeepseekV4Model.__call__'s pipeline_rank != 0 branch:
            # every non-final rank ALSO all_gathers afterward. Whether that
            # all_gather is a no-op for gradient purposes on this rank is
            # exactly what's under test.
            gathered = mx.distributed.all_gather(h_sent)[: h.shape[0]]
            return mx.mean(gathered)  # scalar so value_and_grad is well-formed

        # nn.value_and_grad expects a Module; use mx.value_and_grad on a
        # plain dict of arrays instead, which is the closer analogue to how
        # DWQ trains lm.trainable_parameters().
        lg = mx.value_and_grad(loss_fn)
        loss_val, grads = lg(params)
        mx.eval(loss_val, grads)
        print(f"[rank1] loss={loss_val.item():.6f}", flush=True)
        print(f"[rank1] dW1 grad is None: {grads['W1'] is None}", flush=True)
        if grads["W1"] is not None:
            print(f"[rank1] dW1 nonzero: {bool(mx.any(grads['W1'] != 0).item())}", flush=True)
            print(f"[rank1] dW1={grads['W1'].tolist()}", flush=True)
            print(f"[rank1] ref dW1={ref_dW1.tolist()}", flush=True)
    else:
        params = {"W2": W2, "b2": b2}

        def loss_fn(params):
            h_shape = (x.shape[0], W1.shape[1])
            h = mx.distributed.recv_like(mx.zeros(h_shape, dtype=mx.float32), 1)
            y = h @ params["W2"] + params["b2"]
            gathered = mx.distributed.all_gather(y)[: y.shape[0]]
            return mx.mean((gathered - target) ** 2)

        lg = mx.value_and_grad(loss_fn)
        loss_val, grads = lg(params)
        mx.eval(loss_val, grads)
        print(f"[rank0] loss={loss_val.item():.6f} ref_loss={ref_loss.item():.6f}", flush=True)
        print(f"[rank0] dW2 grad is None: {grads['W2'] is None}", flush=True)
        if grads["W2"] is not None:
            print(f"[rank0] dW2 nonzero: {bool(mx.any(grads['W2'] != 0).item())}", flush=True)
            print(f"[rank0] dW2={grads['W2'].tolist()}", flush=True)
            print(f"[rank0] ref dW2={ref_dW2.tolist()}", flush=True)


if __name__ == "__main__":
    run()
