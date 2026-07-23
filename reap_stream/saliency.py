from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LayerSaliency:
    num_experts: int
    # REAP: running sum of (g * ||f||) and counts over tokens where expert is selected
    reap_sum: np.ndarray = field(init=False)
    reap_count: np.ndarray = field(init=False)
    freq: np.ndarray = field(init=False)
    gate_sum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        e = self.num_experts
        self.reap_sum = np.zeros(e, dtype=np.float64)
        self.reap_count = np.zeros(e, dtype=np.int64)
        self.freq = np.zeros(e, dtype=np.int64)
        self.gate_sum = np.zeros(e, dtype=np.float64)

    def update(
        self,
        expert_ids: np.ndarray,
        gate_weights: np.ndarray,
        activation_norms: np.ndarray,
    ) -> None:
        """Accumulate one batch.

        expert_ids: (tokens, k)
        gate_weights: (tokens, k)
        activation_norms: (tokens, k)
        """
        flat_ids = expert_ids.reshape(-1).astype(np.int64)
        flat_g = gate_weights.reshape(-1).astype(np.float64)
        flat_n = activation_norms.reshape(-1).astype(np.float64)
        contrib = flat_g * flat_n
        for idx, g, c in zip(flat_ids, flat_g, contrib):
            self.freq[idx] += 1
            self.gate_sum[idx] += g
            self.reap_sum[idx] += c
            self.reap_count[idx] += 1

    def reap_scores(self) -> np.ndarray:
        scores = np.full(self.num_experts, np.inf, dtype=np.float64)
        mask = self.reap_count > 0
        scores[mask] = self.reap_sum[mask] / self.reap_count[mask]
        # Never-selected experts: treat as least salient (safe to prune first)
        scores[~mask] = -1.0
        return scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "reap": self.reap_scores().tolist(),
            "freq": self.freq.tolist(),
            "gate_sum": self.gate_sum.tolist(),
            "reap_count": self.reap_count.tolist(),
        }

    def to_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "reap_sum": self.reap_sum.tolist(),
            "reap_count": self.reap_count.tolist(),
            "freq": self.freq.tolist(),
            "gate_sum": self.gate_sum.tolist(),
        }

    @classmethod
    def from_checkpoint_dict(cls, data: dict[str, Any]) -> "LayerSaliency":
        obj = cls(num_experts=int(data["num_experts"]))
        obj.reap_sum = np.array(data["reap_sum"], dtype=np.float64)
        obj.reap_count = np.array(data["reap_count"], dtype=np.int64)
        obj.freq = np.array(data["freq"], dtype=np.int64)
        obj.gate_sum = np.array(data["gate_sum"], dtype=np.float64)
        return obj


def build_plan(
    layer_stats: dict[int, LayerSaliency],
    ratio: float,
    min_experts: int = 1,
) -> dict[str, Any]:
    """Keep highest-REAP experts; prune lowest `ratio` fraction per layer."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")

    plan_layers: dict[str, Any] = {}
    for layer_idx, stats in sorted(layer_stats.items()):
        scores = stats.reap_scores()
        n = stats.num_experts
        n_prune = int(n * ratio)
        n_keep = max(min_experts, n - n_prune)
        n_prune = n - n_keep
        # prune lowest scores (never-seen already at -1)
        order = np.argsort(scores)  # ascending
        prune = order[:n_prune].tolist()
        keep = sorted(order[n_prune:].tolist())
        plan_layers[str(layer_idx)] = {
            "num_experts": n,
            "keep": keep,
            "prune": prune,
            "scores": scores.tolist(),
        }

    return {
        "method": "reap",
        "ratio": ratio,
        "min_experts": min_experts,
        "layers": plan_layers,
    }
