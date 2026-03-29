"""
Pure PyTorch fallback for all TurboQuant kernel operations.

Used when Triton is unavailable or when running on CPU for testing.
All functions are numerically equivalent to the Triton versions.
"""
from __future__ import annotations

import math

import torch

from turboquant_vllm.quantizer import (
    QuantizedKey,
    QuantizedValue,
    _unpack_bits,
    _unpack_signs,
)


def mse_score_pytorch(
    query_rot: torch.Tensor,    # (BH, D) — q @ Pi^T, precomputed
    qk: QuantizedKey,           # packed MSE indices + norms
    centroids: torch.Tensor,    # (2^mse_bits,) float32
) -> torch.Tensor:
    """
    Compute MSE attention score: norm * (q_rot · centroids[idx]).

    Returns: (BH, N) float32 logits.
    """
    # qk.mse_packed: (BH, N, packed_d)
    BH, N, _ = qk.mse_packed.shape
    d = query_rot.shape[-1]

    indices = _unpack_bits(qk.mse_packed, qk.mse_bits, d)  # (BH, N, d)
    centroid_vals = centroids[indices]                        # (BH, N, d)

    # q_rot: (BH, D) → broadcast over N
    scores = (query_rot.unsqueeze(1) * centroid_vals).sum(dim=-1)  # (BH, N)
    scores = scores * qk.norm.float()                              # scale by norms
    return scores


def qjl_score_pytorch(
    q_sketch: torch.Tensor,     # (BH, D) — q @ S^T, precomputed
    qk: QuantizedKey,           # packed QJL signs + residual norms
    d: int,
    qjl_scale: float,
    out: torch.Tensor | None = None,  # (BH, N) — add to in-place if provided
) -> torch.Tensor:
    """
    Compute QJL attention score: qjl_scale * res_norm * (q_sketch · signs).

    Returns: (BH, N) float32 logits.
    """
    BH, N, _ = qk.qjl_signs.shape
    signs = _unpack_signs(qk.qjl_signs, d)           # (BH, N, d)
    dot = (q_sketch.unsqueeze(1) * signs).sum(dim=-1)  # (BH, N)
    qjl_scores = qjl_scale * qk.residual_norm.float() * dot

    if out is not None:
        out = out + qjl_scores
        return out
    return qjl_scores


def turboquant_score_pytorch(
    query: torch.Tensor,         # (BH, D)
    qk: QuantizedKey,
    Pi: torch.Tensor,            # (D, D)
    S: torch.Tensor,             # (D, D)
    centroids: torch.Tensor,
    qjl_scale: float,
) -> torch.Tensor:
    """Combined TurboQuant score (MSE + QJL). Returns (BH, N) logits."""
    q_rot = query.float() @ Pi.T      # (BH, D)
    q_sketch = query.float() @ S.T    # (BH, D)
    d = query.shape[-1]

    mse = mse_score_pytorch(q_rot, qk, centroids)
    total = qjl_score_pytorch(q_sketch, qk, d, qjl_scale, out=mse)
    return total


def hybrid_attention_pytorch(
    query: torch.Tensor,           # (BH, 1, D) or (BH, D)
    compressed_keys: QuantizedKey, # (BH, N_compressed, ...) packed
    compressed_vals: QuantizedValue,
    recent_k: torch.Tensor | None, # (BH, N_recent, D) or None
    recent_v: torch.Tensor | None,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    key_bits: int,
    sm_scale: float,
) -> torch.Tensor:
    """
    Full hybrid attention combining TQ-compressed history with exact recent tokens.

    Returns: (BH, D) attention output, float32.
    """
    if query.dim() == 3:
        query = query.squeeze(1)  # (BH, D)

    BH, d = query.shape
    qjl_scale = math.sqrt(math.pi / 2.0) / d

    # TurboQuant scores for compressed history
    tq_scores = turboquant_score_pytorch(query, compressed_keys, Pi, S, centroids, qjl_scale)
    tq_scores = tq_scores * sm_scale  # (BH, N_compressed)

    # Dequantize compressed values
    tq_vals = _dequantize_values_pytorch(compressed_vals)  # (BH, N_compressed, D)

    if recent_k is not None and recent_k.shape[1] > 0:
        # Exact attention for recent tokens
        recent_scores = (query.unsqueeze(1) * recent_k).sum(dim=-1) * sm_scale  # (BH, N_recent)
        all_scores = torch.cat([tq_scores, recent_scores], dim=-1)  # (BH, N_total)
        all_vals = torch.cat([tq_vals, recent_v.float()], dim=1)     # (BH, N_total, D)
    else:
        all_scores = tq_scores
        all_vals = tq_vals

    # Softmax + weighted sum
    weights = torch.softmax(all_scores, dim=-1)  # (BH, N_total)
    out = (weights.unsqueeze(-1) * all_vals).sum(dim=1)  # (BH, D)
    return out


def _dequantize_values_pytorch(qv: QuantizedValue) -> torch.Tensor:
    """Dequantize group-quantized values. Returns (BH, N, D) float32."""
    *batch, d = qv.data.shape
    gs = qv.group_size
    n_groups = d // gs
    v = qv.data.float().reshape(*batch, n_groups, gs)
    scales = qv.scales.float().unsqueeze(-1)  # (..., n_groups, 1)
    zeros = qv.zeros.float().unsqueeze(-1)
    return (v * scales + zeros).reshape(*batch, d)
