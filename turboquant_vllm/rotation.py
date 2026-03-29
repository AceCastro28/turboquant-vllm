"""
Rotation and sketch matrix generation for TurboQuant.

Pi  — random orthogonal matrix (QR decomposition of Gaussian) for MSE stage
S   — i.i.d. Gaussian matrix for QJL residual stage

Both are deterministic given (seed, layer_idx). They are generated once at
model load and stored on GPU for the lifetime of the model.
"""
from __future__ import annotations

import torch


def make_rotation_matrix(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """
    Generate a d×d random orthogonal matrix via QR decomposition.

    Args:
        d: dimension (head_dim)
        seed: integer seed for reproducibility
        device: target device

    Returns:
        Pi: (d, d) float32 orthogonal matrix
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q.to(device)


def make_qjl_matrix(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """
    Generate a d×d i.i.d. N(0,1) matrix for the QJL sketch.

    Args:
        d: dimension (head_dim)
        seed: integer seed for reproducibility
        device: target device

    Returns:
        S: (d, d) float32 Gaussian matrix
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    S = torch.randn(d, d, generator=gen, dtype=torch.float32)
    return S.to(device)


def make_layer_matrices(
    d: int,
    layer_idx: int,
    base_seed: int = 42,
    device: torch.device = torch.device("cuda"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (Pi, S) for a specific layer. Seeds are deterministic:
        Pi seed = base_seed + layer_idx * 7
        S  seed = base_seed + layer_idx * 7 + 1

    Returns:
        (Pi, S): both (d, d) float32 on device
    """
    layer_seed = base_seed + layer_idx * 7
    Pi = make_rotation_matrix(d, layer_seed, device)
    S = make_qjl_matrix(d, layer_seed + 1, device)
    return Pi, S
