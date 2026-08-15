from __future__ import annotations

from .qwen3 import Qwen3ForCausalLM


class MiniMindIMEForCausalLM(Qwen3ForCausalLM):
    """Inference-only MiniMind-IME adapter.

    The final MiniMind-IME v3 checkpoint uses the same dense GQA decoder
    operators as the AIOS Qwen3 backend: QK norm, RoPE, RMSNorm and SwiGLU.
    Keeping a distinct model class makes the artifact contract and ownership
    explicit while reusing the fused runtime operators.
    """
