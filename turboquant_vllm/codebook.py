"""
Codebook management for TurboQuant.

Lloyd-Max codebooks are precomputed for common head dims (64, 96, 128, 576)
and bits (1-4). Unknown (d, bits) pairs are computed on first use and cached
to disk alongside the built-in codebooks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

_CODEBOOK_DIR = Path(__file__).parent / "codebooks"
_CACHE: dict[tuple[int, int], dict] = {}


def _compute_codebook(d: int, bits: int, max_iter: int = 200, tol: float = 1e-12) -> dict:
    """Compute Lloyd-Max codebook via scipy. Only called on cache miss."""
    from scipy import integrate, special

    n_clusters = 2**bits
    log_const = (
        special.gammaln(d / 2.0) - 0.5 * np.log(np.pi) - special.gammaln((d - 1) / 2.0)
    )
    exp_half = (d - 3) / 2.0

    def pdf(x):
        x = np.clip(x, -1 + 1e-15, 1 - 1e-15)
        return np.exp(log_const + exp_half * np.log(1 - x**2))

    def cond_mean(lo, hi):
        num, _ = integrate.quad(lambda x: x * pdf(np.array([x]))[0], lo, hi)
        den, _ = integrate.quad(lambda x: pdf(np.array([x]))[0], lo, hi)
        return num / max(den, 1e-30)

    # Init: quantile midpoints
    x_grid = np.linspace(-1 + 1e-10, 1 - 1e-10, 10000)
    pdf_vals = pdf(x_grid)
    cdf = np.cumsum(pdf_vals) * (x_grid[1] - x_grid[0])
    cdf /= cdf[-1]

    qedges = np.linspace(0, 1, n_clusters + 1)
    centroids = np.array([x_grid[min(np.searchsorted(cdf, (q + qedges[i + 1]) / 2), len(x_grid) - 1)]
                           for i, q in enumerate(qedges[:-1])])

    prev_cost = float("inf")
    for _ in range(max_iter):
        bounds = np.concatenate([[-1.0], (centroids[:-1] + centroids[1:]) / 2, [1.0]])
        new_c = np.array([cond_mean(bounds[i], bounds[i + 1]) for i in range(n_clusters)])
        cost = sum(
            integrate.quad(lambda x, c=new_c[i], lo=bounds[i], hi=bounds[i + 1]:
                           (x - c)**2 * pdf(np.array([x]))[0], bounds[i], bounds[i + 1])[0]
            for i in range(n_clusters)
        )
        centroids = new_c
        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost

    bounds = np.concatenate([[-1.0], (centroids[:-1] + centroids[1:]) / 2, [1.0]])
    return {
        "centroids": centroids.tolist(),
        "boundaries": bounds.tolist(),
        "mse_per_coord": float(cost),
        "d": d,
        "bits": bits,
    }


def get_codebook(d: int, bits: int) -> dict:
    """Return codebook dict for (d, bits). Computes and caches on miss."""
    key = (d, bits)
    if key in _CACHE:
        return _CACHE[key]

    path = _CODEBOOK_DIR / f"codebook_d{d}_b{bits}.json"
    if path.exists():
        cb = json.loads(path.read_text())
        _CACHE[key] = cb
        return cb

    import logging
    logging.getLogger("turboquant_vllm").info(
        f"[TurboQuant] Computing Lloyd-Max codebook d={d} bits={bits} (one-time)..."
    )
    cb = _compute_codebook(d, bits)
    path.write_text(json.dumps(cb, indent=2))
    _CACHE[key] = cb
    return cb


def get_codebook_tensors(
    d: int,
    bits: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids, boundaries) as GPU tensors."""
    cb = get_codebook(d, bits)
    centroids = torch.tensor(cb["centroids"], device=device, dtype=dtype)
    boundaries = torch.tensor(cb["boundaries"][1:-1], device=device, dtype=dtype)  # inner boundaries only
    return centroids, boundaries
