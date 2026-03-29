"""
TQConfig — single source of truth for all TurboQuant parameters.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os

# Built-in codebook directory
_CODEBOOK_DIR = Path(__file__).parent / "codebooks"


@dataclass
class TQConfig:
    # Key quantization: 3 bits = 2-bit MSE + 1-bit QJL residual
    key_bits: int = 3
    # Value quantization: 2-bit group quantization (standard)
    value_bits: int = 2
    value_group_size: int = 32

    # Ring buffer: exact attention for the most recent N tokens
    ring_capacity: int = 128

    # Mode: "capture_only" (compress but use flash for attn) or "hybrid" (TQ decode)
    mode: str = "hybrid"

    # First N layers use higher precision (initial layers have outsized importance)
    initial_layers_count: int = 4
    initial_layers_key_bits: int = 4  # one bit higher than key_bits

    # Codebook directory (None = use built-in)
    codebook_dir: Path | None = None

    # Random seed for rotation matrices (deterministic per layer via seed + layer_idx * 7)
    seed: int = 42

    def __post_init__(self):
        assert self.mode in ("capture_only", "hybrid"), f"Unknown mode: {self.mode}"
        assert 1 <= self.key_bits <= 4, "key_bits must be 1-4"
        assert 1 <= self.value_bits <= 4, "value_bits must be 1-4"
        if self.codebook_dir is None:
            self.codebook_dir = _CODEBOOK_DIR

    @classmethod
    def from_env(cls) -> "TQConfig":
        """Build config from environment variables (for plugin activation)."""
        return cls(
            key_bits=int(os.environ.get("TQ_KEY_BITS", 3)),
            value_bits=int(os.environ.get("TQ_VALUE_BITS", 2)),
            ring_capacity=int(os.environ.get("TQ_RING_CAPACITY", 128)),
            mode=os.environ.get("TQ_MODE", "hybrid"),
            initial_layers_count=int(os.environ.get("TQ_INITIAL_LAYERS", 4)),
            seed=int(os.environ.get("TQ_SEED", 42)),
        )
