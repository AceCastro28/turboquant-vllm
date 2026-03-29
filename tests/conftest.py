"""Shared fixtures for TurboQuant vLLM tests."""
import pytest
import torch

from turboquant_vllm.config import TQConfig
from turboquant_vllm.rotation import make_layer_matrices
from turboquant_vllm.codebook import get_codebook_tensors


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def config():
    return TQConfig(key_bits=3, value_bits=2, ring_capacity=128, mode="hybrid")


@pytest.fixture
def matrices_d128(device):
    Pi, S = make_layer_matrices(128, layer_idx=0, base_seed=42, device=device)
    return Pi, S


@pytest.fixture
def codebook_d128_b2(device):
    return get_codebook_tensors(128, 2, device)


@pytest.fixture
def sample_keys_d128(device):
    """Random keys shaped like (num_kv_heads, num_tokens, head_dim)."""
    torch.manual_seed(0)
    return torch.randn(8, 64, 128, device=device)


@pytest.fixture
def sample_values_d128(device):
    torch.manual_seed(1)
    return torch.randn(8, 64, 128, device=device)
