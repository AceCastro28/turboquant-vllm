"""
Version-stable hook installer for vLLM attention layers.

Installs TurboQuant intercept on every attention layer in the model runner's
static forward context. Supports vLLM 0.16 (FlashInfer, forward-capture mode)
and vLLM 0.17+ (do_kv_cache_update mode).

Clean API:
    states = install_hooks(model_runner, config)
    uninstall_hooks(model_runner)   # restores all originals
"""
from __future__ import annotations

import logging
import math
import types
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional

import torch
import torch.nn.functional as F

from turboquant_vllm.config import TQConfig
from turboquant_vllm.codebook import get_codebook_tensors
from turboquant_vllm.rotation import make_layer_matrices
from turboquant_vllm.quantizer import TurboQuantizer, QuantizedKey, QuantizedValue

log = logging.getLogger("turboquant_vllm")


# ── Per-request KV store ───────────────────────────────────────────────

@dataclass
class LayerKVStore:
    """Compressed KV store for one layer, one request (or pooled)."""
    keys: Optional[QuantizedKey] = None       # (num_kv_heads, N, ...)
    values: Optional[QuantizedValue] = None   # (num_kv_heads, N, ...)
    num_tokens: int = 0

    # Ring buffer for recent exact tokens
    ring_k: Optional[torch.Tensor] = None  # (num_kv_heads, ring_capacity, D)
    ring_v: Optional[torch.Tensor] = None
    ring_head: int = 0
    ring_size: int = 0

    def reset(self):
        self.keys = None
        self.values = None
        self.num_tokens = 0
        self.ring_k = None
        self.ring_v = None
        self.ring_head = 0
        self.ring_size = 0


# ── Per-layer state ────────────────────────────────────────────────────

@dataclass
class LayerState:
    """All TurboQuant state for one attention layer."""
    layer_idx: int
    head_dim: int
    num_kv_heads: int
    num_query_heads: int
    config: TQConfig
    quantizer: TurboQuantizer
    # Shared KV store (single-request mode; multi-request needs cache_manager)
    store: LayerKVStore = field(default_factory=LayerKVStore)

    # Saved original methods for clean uninstall
    _orig_forward: Optional[object] = None
    _orig_kv_update: Optional[object] = None


# ── Attention patching ─────────────────────────────────────────────────

def _prefill_exact(q, k, v, head_dim, num_query_heads, num_kv_heads, scale):
    """Exact attention for prefill using SDPA (dispatches to efficient CUDA attention)."""
    T = q.shape[0]
    q3 = q.view(T, num_query_heads, head_dim) if q.dim() == 2 else q
    k3 = k.view(T, num_kv_heads, head_dim) if k.dim() == 2 else k
    v3 = v.view(T, num_kv_heads, head_dim) if v.dim() == 2 else v

    if num_query_heads != num_kv_heads:
        reps = num_query_heads // num_kv_heads
        k3 = k3.repeat_interleave(reps, dim=1)
        v3 = v3.repeat_interleave(reps, dim=1)

    q_t = q3.unsqueeze(0).transpose(1, 2)  # (1, H, T, D)
    k_t = k3.unsqueeze(0).transpose(1, 2)
    v_t = v3.unsqueeze(0).transpose(1, 2)

    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    return out.squeeze(0).transpose(0, 1).reshape(T, num_query_heads * head_dim)


def _ingest_kv(state: LayerState, key: torch.Tensor, value: torch.Tensor, num_tokens: int):
    """Compress and store KV tensors into the layer's store."""
    q = state.quantizer
    store = state.store

    # key/value shape from vLLM: (num_tokens, num_kv_heads, head_dim) or (num_tokens, ...)
    k = key[:num_tokens]
    v = value[:num_tokens]

    # Reshape to (num_kv_heads, num_tokens, head_dim) for per-head quantization
    if k.dim() == 2:
        k = k.view(num_tokens, state.num_kv_heads, state.head_dim)
        v = v.view(num_tokens, state.num_kv_heads, state.head_dim)

    k = k.permute(1, 0, 2).contiguous()  # (H_kv, T, D)
    v = v.permute(1, 0, 2).contiguous()

    qk = q.quantize_keys(k)
    qv = q.quantize_values(v)

    if store.keys is None:
        store.keys = qk
        store.values = qv
    else:
        # Concatenate along token dimension (dim=1 of each tensor)
        def cat(a, b):
            return torch.cat([a, b], dim=1)
        store.keys = QuantizedKey(
            mse_packed=cat(store.keys.mse_packed, qk.mse_packed),
            qjl_signs=cat(store.keys.qjl_signs, qk.qjl_signs),
            norm=cat(store.keys.norm, qk.norm),
            residual_norm=cat(store.keys.residual_norm, qk.residual_norm),
            mse_bits=qk.mse_bits,
        )
        store.values = QuantizedValue(
            data=cat(store.values.data, qv.data),
            scales=cat(store.values.scales, qv.scales),
            zeros=cat(store.values.zeros, qv.zeros),
            bits=qv.bits,
            group_size=qv.group_size,
        )

    store.num_tokens += num_tokens


def _decode_hybrid(state: LayerState, query: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """
    Hybrid decode: TurboQuant for compressed history + exact for recent ring buffer.
    Returns (num_query_heads * head_dim,) float tensor matching query dtype.
    """
    store = state.store
    q = state.quantizer

    if store.keys is None or store.num_tokens == 0:
        return None  # No compressed cache yet, fall through to flash

    # Flatten query to (H_q, 1, D) for per-head scoring
    num_tokens = query.shape[0]
    H_q = state.num_query_heads
    H_kv = state.num_kv_heads
    D = state.head_dim

    if query.dim() == 2:
        query_heads = query.view(num_tokens, H_q, D).permute(1, 0, 2)  # (H_q, T, D)
    else:
        query_heads = query.permute(1, 0, 2)

    # GQA: expand KV heads to match query heads
    reps = H_q // H_kv

    from turboquant_vllm.kernels.fallback import hybrid_attention_pytorch
    out_heads = hybrid_attention_pytorch(
        query=query_heads.squeeze(1) if query_heads.shape[1] == 1 else query_heads[:, 0],
        compressed_keys=store.keys,
        compressed_vals=store.values,
        recent_k=None,  # TODO: ring buffer
        recent_v=None,
        Pi=q.Pi,
        S=q.S,
        centroids=q.centroids,
        key_bits=q.key_bits,
        sm_scale=sm_scale,
    )  # (H_q, D)

    return out_heads.reshape(H_q * D).to(query.dtype)


def _make_forward_patch(state: LayerState, orig_forward, capture_in_forward: bool = False):
    """Create patched forward method."""

    @wraps(orig_forward)
    def patched(self_impl, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        mode = state.config.mode

        # Profiling/capture pass: always passthrough
        if attn_metadata is None:
            return orig_forward(self_impl, layer, query, key, value, kv_cache,
                                attn_metadata, output, output_scale, output_block_scale)

        # Capture K/V when no separate do_kv_cache_update exists
        if capture_in_forward and mode != "off":
            num_tokens = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
            _ingest_kv(state, key, value, num_tokens)

        is_prefill = getattr(attn_metadata, "max_query_len", 2) > 1
        if is_prefill or mode == "capture_only":
            return orig_forward(self_impl, layer, query, key, value, kv_cache,
                                attn_metadata, output, output_scale, output_block_scale)

        # Hybrid decode
        scale = getattr(self_impl, "scale", 1.0 / math.sqrt(state.head_dim))
        result = _decode_hybrid(state, query, scale)
        if result is None:
            return orig_forward(self_impl, layer, query, key, value, kv_cache,
                                attn_metadata, output, output_scale, output_block_scale)

        num_actual = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
        if output is not None:
            output[:num_actual] = result[:num_actual * result.shape[-1] // num_actual].reshape(num_actual, -1)
            return output
        return result.unsqueeze(0) if query.dim() == 3 else result

    return patched


def _make_kv_update_patch(state: LayerState, orig_kv_update):
    """Create patched do_kv_cache_update method."""

    @wraps(orig_kv_update)
    def patched(self_impl, layer, key, value, kv_cache, slot_mapping):
        orig_kv_update(self_impl, layer, key, value, kv_cache, slot_mapping)
        num_tokens = slot_mapping.shape[0]
        _ingest_kv(state, key, value, num_tokens)

    return patched


# ── Public API ─────────────────────────────────────────────────────────

def install_hooks(model_runner, config: TQConfig | None = None) -> dict[str, LayerState]:
    """
    Install TurboQuant hooks on all attention layers.

    Args:
        model_runner: vLLM GPU model runner
        config: TQConfig (defaults to TQConfig() if None)

    Returns:
        dict mapping layer_name → LayerState
    """
    if config is None:
        config = TQConfig()

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device

    states: dict[str, LayerState] = {}
    layer_idx = 0

    for layer_name, attn_module in static_ctx.items():
        impl = getattr(attn_module, "impl", None)
        if impl is None:
            continue

        num_kv_heads = getattr(impl, "num_kv_heads", None)
        if num_kv_heads is None:
            continue

        head_dim = getattr(impl, "head_size", None) or getattr(impl, "kv_lora_rank", None)
        if head_dim is None:
            continue

        head_dim = int(head_dim)
        num_kv_heads = int(num_kv_heads)
        num_query_heads = int(
            getattr(attn_module, "num_heads", None)
            or getattr(attn_module, "num_attention_heads", None)
            or getattr(impl, "num_heads", num_kv_heads)
        )

        key_bits = (
            config.initial_layers_key_bits
            if layer_idx < config.initial_layers_count
            else config.key_bits
        )

        # Build rotation matrices and quantizer
        Pi, S = make_layer_matrices(head_dim, layer_idx, config.seed, device)
        centroids, boundaries = get_codebook_tensors(head_dim, key_bits - 1, device)

        quantizer = TurboQuantizer(
            d=head_dim,
            Pi=Pi,
            S=S,
            centroids=centroids,
            boundaries=boundaries,
            key_bits=key_bits,
            value_bits=config.value_bits,
            value_group_size=config.value_group_size,
        )

        state = LayerState(
            layer_idx=layer_idx,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            config=config,
            quantizer=quantizer,
        )

        # Skip MLA (DeepSeek) — not supported yet
        if hasattr(impl, "kv_lora_rank") and not hasattr(impl, "head_size"):
            log.info(f"[TurboQuant] Skipping MLA layer: {layer_name}")
            layer_idx += 1
            continue

        has_kv_update = hasattr(impl, "do_kv_cache_update")

        # Save originals
        state._orig_forward = impl.forward.__func__ if hasattr(impl.forward, "__func__") else impl.forward
        if has_kv_update:
            state._orig_kv_update = impl.do_kv_cache_update.__func__ if hasattr(impl.do_kv_cache_update, "__func__") else impl.do_kv_cache_update

        # Install forward patch
        orig_fwd = state._orig_forward
        patched_fwd = _make_forward_patch(state, orig_fwd, capture_in_forward=not has_kv_update)
        impl.forward = types.MethodType(patched_fwd, impl)

        # Install KV update patch (vLLM 0.17+)
        if has_kv_update:
            orig_kvu = state._orig_kv_update
            patched_kvu = _make_kv_update_patch(state, orig_kvu)
            impl.do_kv_cache_update = types.MethodType(patched_kvu, impl)

        impl._tq_state = state
        states[layer_name] = state

        if layer_idx == 0:
            log.info(
                f"[TurboQuant] Installing hooks: mode={config.mode}, "
                f"key_bits={config.key_bits}, value_bits={config.value_bits}, "
                f"vLLM capture_in_forward={not has_kv_update}"
            )

        layer_idx += 1

    model_runner._tq_states = states
    log.info(f"[TurboQuant] Hooked {len(states)} attention layers.")
    return states


def uninstall_hooks(model_runner) -> None:
    """Restore all original methods. Clean uninstall."""
    states = getattr(model_runner, "_tq_states", {})
    static_ctx = model_runner.compilation_config.static_forward_context

    for layer_name, state in states.items():
        attn_module = static_ctx.get(layer_name)
        if attn_module is None:
            continue
        impl = getattr(attn_module, "impl", None)
        if impl is None:
            continue

        if state._orig_forward is not None:
            impl.forward = types.MethodType(state._orig_forward, impl)
        if state._orig_kv_update is not None:
            impl.do_kv_cache_update = types.MethodType(state._orig_kv_update, impl)

        if hasattr(impl, "_tq_state"):
            del impl._tq_state

    model_runner._tq_states = {}
    log.info(f"[TurboQuant] Uninstalled hooks from {len(states)} layers.")


def reset_kv_stores(model_runner) -> None:
    """Clear all compressed KV stores (call between requests in single-request mode)."""
    for state in getattr(model_runner, "_tq_states", {}).values():
        state.store.reset()
