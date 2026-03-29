"""
vLLM plugin entry point for TurboQuant.

Called by vllm-patched after model load when --kv-cache-dtype turboquant is set.
Also supports direct Python API: turboquant_vllm.activate(model_runner).
"""
from __future__ import annotations

import logging
import os

from turboquant_vllm.config import TQConfig
from turboquant_vllm.hook_installer import install_hooks, uninstall_hooks, reset_kv_stores

log = logging.getLogger("turboquant_vllm")


def activate(model_runner, config: TQConfig | None = None) -> dict:
    """
    Activate TurboQuant on a vLLM model runner.

    Args:
        model_runner: vLLM GPU model runner (post model load)
        config: TQConfig or None (reads from env vars if None)

    Returns:
        layer_states dict
    """
    if config is None:
        config = TQConfig.from_env()

    log.info(
        f"[TurboQuant] Activating — mode={config.mode}, "
        f"key_bits={config.key_bits}, value_bits={config.value_bits}, "
        f"ring_capacity={config.ring_capacity}"
    )

    states = install_hooks(model_runner, config)
    model_runner._tq_config = config
    return states


def deactivate(model_runner) -> None:
    """Remove TurboQuant hooks and restore original vLLM behavior."""
    uninstall_hooks(model_runner)


def get_stats(model_runner) -> dict:
    """Return summary statistics across all TQ layers."""
    states = getattr(model_runner, "_tq_states", {})
    if not states:
        return {}

    total_tokens = 0
    total_layers = len(states)
    config = getattr(model_runner, "_tq_config", TQConfig())

    for state in states.values():
        total_tokens += state.store.num_tokens

    avg_tokens = total_tokens // max(total_layers, 1)
    return {
        "num_layers": total_layers,
        "avg_compressed_tokens_per_layer": avg_tokens,
        "mode": config.mode,
        "key_bits": config.key_bits,
        "value_bits": config.value_bits,
    }
