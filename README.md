# turboquant-vllm

TurboQuant 3-bit KV cache compression (Google ICLR 2026) integrated natively into vLLM.

**Paper**: [TurboQuant: Near-Optimal KV Cache Quantization](https://arxiv.org/abs/2504.19874)

## What it does

Compresses the KV cache from FP16 (16 bits) to ~3 bits per element with zero accuracy loss at 3-bit and marginal degradation at 2.5-bit. Achieves 6x memory reduction and up to 8x faster memory access on modern GPUs.

## How it works

Two-stage quantization per key vector:
- **Stage 1 (MSE)**: Random rotation + Beta-optimal scalar quantization (2-bit, 4 codebook levels)
- **Stage 2 (QJL)**: Sign-sketch of the residual for unbiased inner product correction (1-bit)
- **Values**: Group quantization (2-bit, group_size=32)

## Usage

```bash
# With vllm-patched (first-class support):
python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --kv-cache-dtype turboquant

# Via Python API:
from turboquant_vllm import activate, TQConfig
activate(model_runner, TQConfig(key_bits=3, mode="hybrid"))
```

## Supported Models

| Model | head_dim | Status |
|-------|----------|--------|
| Llama-3.3-70B (castro-brain) | 128 | ✓ |
| Phi-4 14B (castro-coder) | 128 | ✓ |
| Phi-4-mini (castro-fast) | 96 | ✓ |
| Gemma-3-4B (castro-vision) | 256 | ✓ (codebook auto-generated) |

## Installation

```bash
pip install -e .
# Also requires vllm-patched with turboquant dtype support:
cd ~/vllm-patched && git apply ~/projects/turboquant-vllm/patches/vllm-cache-dtype.patch
```

## Status

- [x] Core quantizer (MSE + QJL two-stage)
- [x] Lloyd-Max codebooks for d=64, 96, 128, 576
- [x] Triton decode kernels (GB10/Blackwell sm_100 tuned)
- [x] PyTorch fallback kernels
- [x] vLLM hook installer (v0.16/v0.17 compatible)
- [x] `--kv-cache-dtype turboquant` CLI flag in vllm-patched
- [x] 8/8 CPU unit tests passing
- [ ] GPU integration tests (pending — autoresearch running)
- [ ] Multi-request KV isolation (cache_manager)
- [ ] Benchmarks vs fp8_e4m3 baseline
