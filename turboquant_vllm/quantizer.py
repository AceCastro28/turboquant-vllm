"""
TurboQuant two-stage quantizer.

Stage 1 (MSE): Random rotation + Beta-optimal scalar quantization
Stage 2 (QJL): Sign-sketch of the residual for unbiased inner product correction

Memory layout per token key (d=128, key_bits=3):
    mse_packed   : (d//vals_per_byte,) uint8  — (b-1)-bit indices, packed
    qjl_signs    : (d//8,) uint8              — 1 sign bit per coordinate
    norm         : scalar float16             — original L2 norm
    residual_norm: scalar float16             — residual L2 norm after MSE

Value quantization is simple group quantization (not TurboQuant).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


# ── Bit packing helpers ────────────────────────────────────────────────

def _pack_bits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack integer indices into uint8 tensors.
    indices: (..., d) int32, values in [0, 2^bits)
    Returns: (..., d // vals_per_byte) uint8
    """
    if bits == 1:
        vals_per_byte, eff_bits = 8, 1
    elif bits == 2:
        vals_per_byte, eff_bits = 4, 2
    else:
        # 3-bit rounds up to 4-bit packing: 2 values per byte
        vals_per_byte, eff_bits = 2, 4

    *batch, d = indices.shape
    n_bytes = d // vals_per_byte
    idx = indices.to(torch.int32).view(*batch, n_bytes, vals_per_byte)
    packed = torch.zeros(*batch, n_bytes, dtype=torch.uint8, device=indices.device)
    for i in range(vals_per_byte):
        packed |= ((idx[..., i] & ((1 << eff_bits) - 1)) << (i * eff_bits)).to(torch.uint8)
    return packed


def _unpack_bits(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """
    Unpack uint8 tensor back to integer indices.
    packed: (..., d // vals_per_byte) uint8
    Returns: (..., d) int32
    """
    if bits == 1:
        vals_per_byte, eff_bits = 8, 1
    elif bits == 2:
        vals_per_byte, eff_bits = 4, 2
    else:
        vals_per_byte, eff_bits = 2, 4

    *batch, n_bytes = packed.shape
    packed_i = packed.to(torch.int32)
    chunks = []
    mask = (1 << eff_bits) - 1
    for i in range(vals_per_byte):
        chunks.append((packed_i >> (i * eff_bits)) & mask)
    # chunks: list of (..., n_bytes) → interleave
    out = torch.stack(chunks, dim=-1).view(*batch, n_bytes * vals_per_byte)
    return out[..., :d]


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """
    Pack +1/-1 sign tensor into uint8 bit vector.
    signs: (..., d) float, values in {-1, +1}
    Returns: (..., d // 8) uint8
    """
    *batch, d = signs.shape
    bits = ((signs > 0).to(torch.int32))  # 1 for +1, 0 for -1
    n_bytes = math.ceil(d / 8)
    # Pad to multiple of 8
    if d % 8 != 0:
        pad = 8 - d % 8
        bits = F.pad(bits, (0, pad))
    bits = bits.view(*batch, n_bytes, 8)
    packed = torch.zeros(*batch, n_bytes, dtype=torch.uint8, device=signs.device)
    for i in range(8):
        packed |= (bits[..., i] << i).to(torch.uint8)
    return packed


def _unpack_signs(packed: torch.Tensor, d: int) -> torch.Tensor:
    """
    Unpack uint8 bit vector back to +1/-1 float signs.
    packed: (..., d // 8) uint8
    Returns: (..., d) float32
    """
    *batch, n_bytes = packed.shape
    packed_i = packed.to(torch.int32)
    bits = torch.stack([(packed_i >> i) & 1 for i in range(8)], dim=-1)
    bits = bits.view(*batch, n_bytes * 8)[..., :d]
    return bits.float() * 2 - 1  # {0,1} → {-1, +1}


# ── Quantized key namedtuple ───────────────────────────────────────────

class QuantizedKey(NamedTuple):
    mse_packed: torch.Tensor     # (..., d // vals_per_byte) uint8
    qjl_signs: torch.Tensor      # (..., d // 8) uint8
    norm: torch.Tensor           # (...,) float16
    residual_norm: torch.Tensor  # (...,) float16
    mse_bits: int                # bits used for MSE stage (key_bits - 1)


class QuantizedValue(NamedTuple):
    data: torch.Tensor    # (..., d // vals_per_byte) uint8
    scales: torch.Tensor  # (..., n_groups) float16
    zeros: torch.Tensor   # (..., n_groups) float16
    bits: int
    group_size: int


# ── Main quantizer ─────────────────────────────────────────────────────

class TurboQuantizer:
    """
    Stateful per-layer TurboQuant quantizer.

    Holds the rotation matrix Pi and QJL matrix S for a single layer/head.
    Thread-safe for inference (no mutable state after init).
    """

    def __init__(
        self,
        d: int,
        Pi: torch.Tensor,     # (d, d) rotation matrix
        S: torch.Tensor,      # (d, d) QJL sketch matrix
        centroids: torch.Tensor,   # (2^mse_bits,) codebook centroids
        boundaries: torch.Tensor,  # (2^mse_bits - 1,) inner boundaries
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
    ):
        self.d = d
        self.Pi = Pi
        self.S = S
        self.centroids = centroids
        self.boundaries = boundaries
        self.key_bits = key_bits
        self.mse_bits = key_bits - 1  # MSE uses (key_bits - 1), QJL uses 1
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self._qjl_scale = math.sqrt(math.pi / 2.0) / d

    # ── Key quantization ───────────────────────────────────────────────

    def quantize_keys(self, keys: torch.Tensor) -> QuantizedKey:
        """
        Quantize key vectors using TurboQuant two-stage algorithm.

        Args:
            keys: (..., d) float tensor of key vectors

        Returns:
            QuantizedKey namedtuple
        """
        orig_dtype = keys.dtype
        keys_f = keys.float()

        # Store original norms
        norms = keys_f.norm(dim=-1)  # (...,)
        norms_safe = norms.clamp(min=1e-8)

        # Normalize to unit sphere
        keys_unit = keys_f / norms_safe.unsqueeze(-1)  # (..., d)

        # Stage 1: Rotate + MSE quantize
        rotated = keys_unit @ self.Pi.T  # (..., d)

        # Scalar quantize each coordinate: find codebook index via searchsorted
        # boundaries shape: (2^mse_bits - 1,) — inner boundaries
        *batch, d = rotated.shape
        flat = rotated.reshape(-1, d)  # (N, d)
        # searchsorted: for each element, find how many boundaries it exceeds
        # boundaries are sorted ascending, result is index in [0, 2^mse_bits)
        indices = torch.searchsorted(
            self.boundaries.unsqueeze(0).expand(flat.shape[0], -1).contiguous(),
            flat.contiguous(),
        )  # (N, d), values in [0, 2^mse_bits)
        indices = indices.reshape(*batch, d)

        # Pack MSE indices
        mse_packed = _pack_bits(indices, self.mse_bits)  # (..., d // vpb) uint8

        # Compute MSE dequantized version to get residual
        dequant_rotated = self.centroids[indices]  # (..., d)
        dequant_unit = dequant_rotated @ self.Pi   # rotate back, (..., d)
        dequant_keys = dequant_unit * norms_safe.unsqueeze(-1)

        # Stage 2: QJL on residual
        residual = keys_f - dequant_keys  # (..., d)
        residual_norms = residual.norm(dim=-1)  # (...,)
        residual_norms_safe = residual_norms.clamp(min=1e-8)
        residual_unit = residual / residual_norms_safe.unsqueeze(-1)

        # QJL sketch: sign(S · residual_unit)
        sketch = residual_unit @ self.S.T  # (..., d)
        signs = torch.sign(sketch)  # (..., d), values in {-1, 0, +1}
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)  # map 0 → +1
        qjl_signs = _pack_signs(signs)  # (..., d//8) uint8

        return QuantizedKey(
            mse_packed=mse_packed,
            qjl_signs=qjl_signs,
            norm=norms.to(torch.float16),
            residual_norm=residual_norms.to(torch.float16),
            mse_bits=self.mse_bits,
        )

    def dequantize_keys(self, qk: QuantizedKey) -> torch.Tensor:
        """
        Dequantize keys. Used for validation/debugging — not on the inference hot path.

        Returns: (..., d) float32
        """
        # Unpack MSE indices → lookup centroids → rotate back
        indices = _unpack_bits(qk.mse_packed, qk.mse_bits, self.d)  # (..., d) int32
        dequant_rotated = self.centroids[indices]  # (..., d) float32
        dequant_unit = dequant_rotated @ self.Pi  # (..., d)
        norms = qk.norm.float()
        dequant_keys = dequant_unit * norms.unsqueeze(-1)

        # Add QJL residual correction
        signs = _unpack_signs(qk.qjl_signs, self.d)  # (..., d) float32
        # Dequant residual: (sqrt(pi/2) / d) * residual_norm * S^T @ signs
        res_norms = qk.residual_norm.float()
        # Inner product estimator: correction = qjl_scale * res_norm * S^T · signs
        # For dequantization we reconstruct the residual estimate
        residual_est = (self._qjl_scale * res_norms.unsqueeze(-1)) * (signs @ self.S)

        return dequant_keys + residual_est

    def compute_attention_scores(
        self,
        query: torch.Tensor,   # (..., d)
        qk: QuantizedKey,
    ) -> torch.Tensor:
        """
        Compute <query, dequant(key)> efficiently without full dequantization.

        Stage 1: q_rot · centroids[idx] * norm  (MSE contribution)
        Stage 2: qjl_scale * res_norm * (q @ S^T) · signs  (QJL contribution)

        Returns: (...,) attention logits (before softmax scaling)
        """
        q_f = query.float()

        # Precompute rotated query (one matmul per decode step)
        q_rot = q_f @ self.Pi.T  # (..., d)

        # Stage 1: MSE score
        indices = _unpack_bits(qk.mse_packed, qk.mse_bits, self.d)
        centroid_vals = self.centroids[indices]  # (..., d)
        mse_score = (q_rot * centroid_vals).sum(dim=-1)  # (...,)
        mse_score = mse_score * qk.norm.float()

        # Stage 2: QJL score
        q_sketch = q_f @ self.S.T  # (..., d)
        signs = _unpack_signs(qk.qjl_signs, self.d)  # (..., d)
        qjl_dot = (q_sketch * signs).sum(dim=-1)  # (...,)
        qjl_score = self._qjl_scale * qk.residual_norm.float() * qjl_dot

        return mse_score + qjl_score

    # ── Value quantization (group quantization) ────────────────────────

    def quantize_values(self, values: torch.Tensor) -> QuantizedValue:
        """
        Simple group quantization for value vectors.

        Args:
            values: (..., d) float tensor

        Returns:
            QuantizedValue namedtuple
        """
        *batch, d = values.shape
        gs = min(self.value_group_size, d)
        n_groups = d // gs
        v = values.float().reshape(*batch, n_groups, gs)

        v_max = v.amax(dim=-1, keepdim=True)
        v_min = v.amin(dim=-1, keepdim=True)
        scales = (v_max - v_min) / (2**self.value_bits - 1)
        scales = scales.clamp(min=1e-8)
        zeros = v_min

        n_levels = 2**self.value_bits - 1
        quantized = ((v - zeros) / scales).round().clamp(0, n_levels).to(torch.uint8)
        quantized_flat = quantized.reshape(*batch, d)

        return QuantizedValue(
            data=quantized_flat,
            scales=scales.squeeze(-1).to(torch.float16),
            zeros=zeros.squeeze(-1).to(torch.float16),
            bits=self.value_bits,
            group_size=gs,
        )

    def dequantize_values(self, qv: QuantizedValue) -> torch.Tensor:
        """Dequantize values. Returns (..., d) float32."""
        *batch, d = qv.data.shape
        gs = qv.group_size
        n_groups = d // gs
        v = qv.data.float().reshape(*batch, n_groups, gs)
        scales = qv.scales.float().unsqueeze(-1)
        zeros = qv.zeros.float().unsqueeze(-1)
        return (v * scales + zeros).reshape(*batch, d)
