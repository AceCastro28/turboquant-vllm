"""
turboquant-vllm: TurboQuant 3-bit KV cache compression for vLLM.

Google ICLR 2026 paper implementation.
"""
from turboquant_vllm.config import TQConfig
from turboquant_vllm.plugin import activate, deactivate, get_stats

__version__ = "0.1.0"
__all__ = ["TQConfig", "activate", "deactivate", "get_stats"]
