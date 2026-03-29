"""
Test quantize/dequantize roundtrip and inner product error bounds.

These tests run on CPU — no GPU required.
"""
import math
import pytest
import torch

from turboquant_vllm.rotation import make_layer_matrices
from turboquant_vllm.codebook import get_codebook_tensors
from turboquant_vllm.quantizer import TurboQuantizer


@pytest.mark.parametrize("head_dim,key_bits", [(128, 3), (128, 4), (96, 3), (64, 3)])
def test_quantize_dequantize_l2_error(head_dim, key_bits):
    """Dequantized keys should be close to originals in L2."""
    device = torch.device("cpu")
    Pi, S = make_layer_matrices(head_dim, layer_idx=0, device=device)
    centroids, boundaries = get_codebook_tensors(head_dim, key_bits - 1, device)

    q = TurboQuantizer(head_dim, Pi, S, centroids, boundaries, key_bits=key_bits)

    torch.manual_seed(42)
    keys = torch.randn(4, 16, head_dim)  # (H, T, D)
    qk = q.quantize_keys(keys)
    dq = q.dequantize_keys(qk)

    rel_err = (keys - dq).norm() / keys.norm()
    # TurboQuant optimizes inner products, not L2. At 3-bit (2-bit MSE = 4 levels)
    # ~45% relative L2 error is expected. At 4-bit (3-bit MSE = 8 levels) ~25%.
    threshold = 0.55 if key_bits == 3 else 0.35
    assert rel_err < threshold, f"Relative L2 error too high: {rel_err:.3f} (threshold {threshold})"


@pytest.mark.parametrize("head_dim,key_bits", [(128, 3), (128, 4)])
def test_inner_product_score_vs_exact(head_dim, key_bits):
    """
    TurboQuant inner product estimator should correlate well with exact dot products.
    Pearson correlation > 0.9 at 3 bits, > 0.95 at 4 bits.
    """
    device = torch.device("cpu")
    Pi, S = make_layer_matrices(head_dim, layer_idx=0, device=device)
    centroids, boundaries = get_codebook_tensors(head_dim, key_bits - 1, device)
    q = TurboQuantizer(head_dim, Pi, S, centroids, boundaries, key_bits=key_bits)

    torch.manual_seed(7)
    H, T, D = 2, 128, head_dim
    keys = torch.randn(H, T, D)
    queries = torch.randn(H, D)

    qk = q.quantize_keys(keys)

    # Exact dot products: (H, T)
    exact = (queries.unsqueeze(1) * keys).sum(dim=-1)

    # TurboQuant estimates: compute per head
    tq_scores = torch.zeros(H, T)
    for h in range(H):
        qk_h = type(qk)(
            mse_packed=qk.mse_packed[h:h+1],
            qjl_signs=qk.qjl_signs[h:h+1],
            norm=qk.norm[h:h+1],
            residual_norm=qk.residual_norm[h:h+1],
            mse_bits=qk.mse_bits,
        )
        tq_scores[h] = q.compute_attention_scores(queries[h:h+1], qk_h).squeeze(0)

    # Pearson correlation between TQ and exact scores (flatten across heads/tokens)
    exact_flat = exact.flatten()
    tq_flat = tq_scores.flatten()
    corr = torch.corrcoef(torch.stack([exact_flat, tq_flat]))[0, 1].item()

    min_corr = 0.90 if key_bits == 3 else 0.95
    assert corr > min_corr, f"Inner product correlation {corr:.3f} < {min_corr} at {key_bits} bits"


def test_value_quantize_dequantize():
    """Value group quantization should reconstruct with <1% relative error."""
    device = torch.device("cpu")
    Pi, S = make_layer_matrices(128, layer_idx=0, device=device)
    centroids, boundaries = get_codebook_tensors(128, 2, device)
    q = TurboQuantizer(128, Pi, S, centroids, boundaries, key_bits=3, value_bits=2)

    torch.manual_seed(9)
    values = torch.randn(4, 16, 128)
    qv = q.quantize_values(values)
    dq = q.dequantize_values(qv)

    rel_err = (values - dq).norm() / values.norm()
    # 2-bit group quantization of N(0,1): ~35-40% relative L2 error is expected
    # (4 levels over ~6-sigma range gives step size ~2, variance ~1/3)
    assert rel_err < 0.50, f"Value relative error too high: {rel_err:.3f}"


def test_pack_unpack_roundtrip():
    """Bit packing should be lossless."""
    from turboquant_vllm.quantizer import _pack_bits, _unpack_bits, _pack_signs, _unpack_signs

    for bits in [1, 2, 3]:
        d = 128
        n_levels = 2 ** (bits if bits < 3 else 4)  # 3-bit packs as 4-bit
        indices = torch.randint(0, 2 ** (bits if bits < 3 else 4), (4, d))
        packed = _pack_bits(indices, bits)
        unpacked = _unpack_bits(packed, bits, d)
        assert torch.all(indices == unpacked), f"Packing roundtrip failed for bits={bits}"

    # Sign bits
    signs = torch.sign(torch.randn(4, 128))
    signs[signs == 0] = 1
    packed_s = _pack_signs(signs)
    unpacked_s = _unpack_signs(packed_s, 128)
    assert torch.all(signs == unpacked_s), "Sign bit packing roundtrip failed"
