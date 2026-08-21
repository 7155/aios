from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from aios.attention import FlashInferAttentionBackend
from aios.core import get_global_ctx
from aios.kernel import triton_attnres_mix
from aios.layers import (
    BaseOP,
    Embedding,
    LMHead,
    OPList,
    RMSNorm,
    RotaryEmbedding,
)

from .qwen3 import Qwen3Attention, Qwen3ForCausalLM, Qwen3MLP

if TYPE_CHECKING:
    from .config import ModelConfig
    from aios.core import Batch
    from aios.kvcache import MHAKVCache


def _attnres_scores(
    sources: torch.Tensor,
    weighted_query: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return routing scores without materializing RMS-normalized sources.

    ``sources`` is ``[S,N,D]`` in AIOS: S depth sources, N active tokens and
    D hidden channels. Expanding ``query · RMSNorm(source)`` lets Inductor
    fuse the FP32 reductions while retaining only the small ``[S,N]`` score
    tensor instead of a second ``[S,N,D]`` normalized bank.
    """
    source_float = sources.float()
    inverse_rms = torch.rsqrt(source_float.square().mean(dim=-1) + eps)
    numerator = torch.einsum("d,snd->sn", weighted_query, source_float)
    return numerator * inverse_rms


def _attnres_mix_bank_eager(
    bank: torch.Tensor,
    weighted_query: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    logits = _attnres_scores(bank, weighted_query, eps)
    weights = logits.softmax(dim=0).to(dtype=bank.dtype)
    return torch.einsum("sn,snd->nd", weights, bank)


def _attnres_mix_partial_eager(
    bank: torch.Tensor,
    partial: torch.Tensor,
    weighted_query: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    bank_logits = _attnres_scores(bank, weighted_query, eps)
    partial_float = partial.float()
    partial_inverse_rms = torch.rsqrt(
        partial_float.square().mean(dim=-1) + eps
    )
    partial_logits = torch.einsum(
        "d,nd->n", weighted_query, partial_float
    ) * partial_inverse_rms
    # Only the small depth-score tensor is concatenated. The much larger
    # [S,N,D] value bank and [N,D] partial remain separate.
    logits = torch.cat((bank_logits, partial_logits.unsqueeze(0)), dim=0)
    weights = logits.softmax(dim=0).to(dtype=bank.dtype)
    mixed = torch.einsum("sn,snd->nd", weights[:-1], bank)
    return mixed + weights[-1].unsqueeze(-1) * partial


# Compile the shared operator functions rather than 65 layer-owned modules.
# Query and RMS weights remain ordinary tensor inputs, so all mixers can reuse
# the same graphs. Dynamic shapes cover changing source depth and active rows.
_attnres_mix_bank_compiled: Callable[..., torch.Tensor] = torch.compile(
    _attnres_mix_bank_eager,
    dynamic=True,
    fullgraph=True,
)
_attnres_mix_partial_compiled: Callable[..., torch.Tensor] = torch.compile(
    _attnres_mix_partial_eager,
    dynamic=True,
    fullgraph=True,
)


class BlockAttnResMixer(BaseOP):
    """Inference-only Kimi Block AttnRes depth mixer for ``alpha=1``.

    The checkpoint-compatible weights are ``query`` and
    ``key_norm.weight``. ``reference`` exists only for numerical/performance
    A/B; production uses the fused Triton hot path.
    """

    def __init__(self, hidden_size: int, eps: float, backend: str) -> None:
        self.query = torch.empty(hidden_size)
        self.key_norm = RMSNorm(hidden_size, eps)
        self._backend = backend
        self._weighted_query: torch.Tensor | None = None

    def _routing_query(self) -> torch.Tensor:
        cached = self._weighted_query
        if (
            cached is None
            or cached.device != self.query.device
            or cached.shape != self.query.shape
        ):
            cached = (
                self.query.float() * self.key_norm.weight.float()
            ).contiguous()
            self._weighted_query = cached
        return cached

    def _reference(
        self,
        bank: torch.Tensor,
        partial: torch.Tensor | None,
    ) -> torch.Tensor:
        sources = (
            bank
            if partial is None
            else torch.cat((bank, partial.unsqueeze(0)), dim=0)
        )
        source_float = sources.float()
        inverse_rms = torch.rsqrt(
            source_float.square().mean(dim=-1, keepdim=True)
            + self.key_norm.eps
        )
        normalized = (
            source_float
            * inverse_rms
            * self.key_norm.weight.float()
        )
        logits = torch.einsum(
            "d,snd->sn", self.query.float(), normalized
        )
        weights = logits.softmax(dim=0).to(dtype=sources.dtype)
        return torch.einsum("sn,snd->nd", weights, sources)

    def forward(
        self,
        bank: torch.Tensor,
        partial: torch.Tensor | None = None,
        *,
        score_buffer: torch.Tensor | None = None,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bank.ndim != 3:
            raise ValueError(
                f"AttnRes bank must have shape [S,N,D], got {tuple(bank.shape)}"
            )
        if bank.shape[-1] != self.query.numel():
            raise ValueError("AttnRes bank hidden size does not match query")
        if partial is not None and partial.shape != bank.shape[1:]:
            raise ValueError("AttnRes partial must have shape [N,D] matching bank")

        if self._backend == "reference":
            return self._reference(bank, partial)

        weighted_query = self._routing_query()
        if self._backend == "triton":
            return triton_attnres_mix(
                bank,
                partial,
                weighted_query,
                self.key_norm.eps,
                score_buffer=score_buffer,
                output_buffer=output_buffer,
            )
        if self._backend == "compiled":
            if partial is None:
                return _attnres_mix_bank_compiled(
                    bank, weighted_query, self.key_norm.eps
                )
            return _attnres_mix_partial_compiled(
                bank, partial, weighted_query, self.key_norm.eps
            )

        if partial is None:
            return _attnres_mix_bank_eager(
                bank, weighted_query, self.key_norm.eps
            )
        return _attnres_mix_partial_eager(
            bank, partial, weighted_query, self.key_norm.eps
        )


class MiniMindAttnResDecoderLayer(BaseOP):
    """One attention/MLP pair that contributes deltas to its local block."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        attn_backend: FlashInferAttentionBackend,
    ) -> None:
        self.self_attn = Qwen3Attention(config, layer_idx, attn_backend)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.attn_res_mixer = BlockAttnResMixer(
            config.hidden_size, config.rms_norm_eps, config.attnres_backend
        )
        self.mlp_res_mixer = BlockAttnResMixer(
            config.hidden_size, config.rms_norm_eps, config.attnres_backend
        )

    def forward(
        self,
        bank: torch.Tensor,
        partial: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        paged_kv_cache: MHAKVCache,
        batch: Batch,
        score_buffer: torch.Tensor | None = None,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.attn_res_mixer.forward(
            bank,
            partial,
            score_buffer=score_buffer,
            output_buffer=output_buffer,
        )
        attention_delta = self.self_attn.forward(
            self.input_layernorm.forward(hidden_states),
            position_embeddings,
            paged_kv_cache,
            batch,
        )
        partial = (
            attention_delta
            if partial is None
            else partial + attention_delta
        )

        hidden_states = self.mlp_res_mixer.forward(
            bank,
            partial,
            score_buffer=score_buffer,
            output_buffer=output_buffer,
        )
        mlp_delta = self.mlp.forward(
            self.post_attention_layernorm.forward(hidden_states)
        )
        return partial + mlp_delta


class MiniMindBlockAttnResModel(BaseOP):
    """32-layer inference trunk with block-local deltas and depth routing."""

    def __init__(self, config: ModelConfig) -> None:
        self.attn_backend = FlashInferAttentionBackend(config)
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([
            MiniMindAttnResDecoderLayer(config, index, self.attn_backend)
            for index in range(config.num_layers)
        ])
        self.final_attn_res_mixer = BlockAttnResMixer(
            config.hidden_size, config.rms_norm_eps, config.attnres_backend
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self._num_blocks = config.attnres_num_blocks
        self._layers_per_block = config.layers_per_attnres_block

    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        input_ids = ctx.batch.input_ids
        paged_kv_cache = ctx.kv_cache
        batch = ctx.batch
        hidden_states = self.embed_tokens.forward(input_ids)
        position_embeddings = self._rotary_emb.forward(batch.positions)

        # Preallocate the complete depth bank once. At most nine sources are
        # retained for the 8-block model: embedding plus eight block deltas.
        bank_storage = torch.empty(
            (self._num_blocks + 1, *hidden_states.shape),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        bank_storage[0].copy_(hidden_states)
        score_storage = torch.empty(
            (self._num_blocks + 1, hidden_states.shape[0]),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        mixed_storage = torch.empty_like(hidden_states)
        source_count = 1

        for block_index in range(self._num_blocks):
            partial: torch.Tensor | None = None
            layer_start = block_index * self._layers_per_block
            layer_end = layer_start + self._layers_per_block
            bank = bank_storage[:source_count]
            for layer_index in range(layer_start, layer_end):
                partial = self.layers.op_list[layer_index].forward(
                    bank,
                    partial,
                    position_embeddings,
                    paged_kv_cache,
                    batch,
                    score_storage,
                    mixed_storage,
                )
            if partial is None:
                raise RuntimeError("AttnRes block produced no residual delta")
            bank_storage[source_count].copy_(partial)
            source_count += 1

        hidden_states = self.final_attn_res_mixer.forward(
            bank_storage[:source_count],
            score_buffer=score_storage,
            output_buffer=mixed_storage,
        )
        return self.norm.forward(hidden_states)


class MiniMindIMEForCausalLM(Qwen3ForCausalLM):
    """Backward-compatible MiniMind-IME inference model.

    Legacy 14-layer/0.1B bundles continue through the standard Qwen3 residual
    implementation. New 0.214B bundles select the native Block AttnRes trunk
    without changing attention, RoPE, SwiGLU, KV cache, or LM-head formats.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.residual_type == "standard":
            super().__init__(config)
            return
        self.model = MiniMindBlockAttnResModel(config)
        self.attn_backend = self.model.attn_backend
        self.lm_head = LMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=(
                self.model.embed_tokens if config.tie_word_embeddings else None
            ),
        )

    def forward(self) -> torch.Tensor:
        hidden_states = self.model.forward()
        return self.lm_head.forward(hidden_states)
