from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_tokens"])
def _attnres_score_kernel(
    bank_ptr,
    partial_ptr,
    weighted_query_ptr,
    score_ptr,
    num_tokens,
    eps: tl.constexpr,
    hidden_size: tl.constexpr,
    bank_sources: tl.constexpr,
    has_partial: tl.constexpr,
    block_hidden: tl.constexpr,
):
    source_index = tl.program_id(0)
    token_index = tl.program_id(1)
    hidden_offsets = tl.arange(0, block_hidden)
    hidden_mask = hidden_offsets < hidden_size

    bank_offsets = (
        source_index * num_tokens * hidden_size
        + token_index * hidden_size
        + hidden_offsets
    )
    bank_mask = (source_index < bank_sources) & hidden_mask
    source = tl.load(
        bank_ptr + bank_offsets,
        mask=bank_mask,
        other=0.0,
    ).to(tl.float32)
    if has_partial:
        partial_offsets = token_index * hidden_size + hidden_offsets
        partial_mask = (source_index == bank_sources) & hidden_mask
        source += tl.load(
            partial_ptr + partial_offsets,
            mask=partial_mask,
            other=0.0,
        ).to(tl.float32)

    weighted_query = tl.load(
        weighted_query_ptr + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    square_sum = tl.sum(source * source, axis=0)
    numerator = tl.sum(source * weighted_query, axis=0)
    inverse_rms = tl.rsqrt(square_sum / hidden_size + eps)
    tl.store(
        score_ptr + source_index * num_tokens + token_index,
        numerator * inverse_rms,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _attnres_value_kernel(
    bank_ptr,
    partial_ptr,
    score_ptr,
    output_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    bank_sources: tl.constexpr,
    total_sources: tl.constexpr,
    has_partial: tl.constexpr,
    block_sources: tl.constexpr,
    block_hidden: tl.constexpr,
):
    token_index = tl.program_id(0)
    hidden_block_index = tl.program_id(1)
    source_offsets = tl.arange(0, block_sources)
    hidden_offsets = (
        hidden_block_index * block_hidden + tl.arange(0, block_hidden)
    )
    source_mask = source_offsets < total_sources
    hidden_mask = hidden_offsets < hidden_size

    scores = tl.load(
        score_ptr + source_offsets * num_tokens + token_index,
        mask=source_mask,
        other=-float("inf"),
    ).to(tl.float32)
    scores -= tl.max(scores, axis=0)
    exponentials = tl.exp(scores)
    # Training converts depth-softmax weights back to the model dtype before
    # value aggregation. Preserve that BF16/FP16 rounding contract here; using
    # FP32 weights changes near-tie logits after 65 mixers.
    weights = (exponentials / tl.sum(exponentials, axis=0)).to(
        bank_ptr.dtype.element_ty
    )

    bank_offsets = (
        source_offsets[:, None] * num_tokens * hidden_size
        + token_index * hidden_size
        + hidden_offsets[None, :]
    )
    bank_mask = source_mask[:, None] & (
        source_offsets[:, None] < bank_sources
    ) & hidden_mask[None, :]
    values = tl.load(
        bank_ptr + bank_offsets,
        mask=bank_mask,
        other=0.0,
    ).to(tl.float32)
    if has_partial:
        partial_values = tl.load(
            partial_ptr + token_index * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        values = tl.where(
            source_offsets[:, None] == bank_sources,
            partial_values[None, :],
            values,
        )

    mixed = tl.sum(values * weights[:, None], axis=0)
    tl.store(
        output_ptr + token_index * hidden_size + hidden_offsets,
        mixed,
        mask=hidden_mask,
    )


def triton_attnres_mix(
    bank: torch.Tensor,
    partial: torch.Tensor | None,
    weighted_query: torch.Tensor,
    eps: float,
    *,
    score_buffer: torch.Tensor | None = None,
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FP32 depth routing and BF16 value aggregation in two kernels."""
    if not bank.is_cuda:
        raise ValueError("Triton AttnRes requires CUDA tensors")
    if bank.ndim != 3:
        raise ValueError("bank must have shape [S,N,D]")
    if not bank.is_contiguous():
        raise ValueError("bank must be contiguous for the Triton pointer layout")
    bank_sources, num_tokens, hidden_size = bank.shape
    if bank_sources < 1 or num_tokens < 1 or hidden_size < 1:
        raise ValueError("bank dimensions must be positive")
    total_sources = bank_sources + int(partial is not None)
    if partial is not None and partial.shape != bank.shape[1:]:
        raise ValueError("partial must match bank [N,D]")
    if partial is not None and not partial.is_contiguous():
        raise ValueError("partial must be contiguous")
    if weighted_query.shape != (hidden_size,):
        raise ValueError("weighted_query must match hidden size")
    if not weighted_query.is_contiguous():
        raise ValueError("weighted_query must be contiguous")

    if score_buffer is None:
        scores = torch.empty(
            (total_sources, num_tokens),
            dtype=torch.float32,
            device=bank.device,
        )
    else:
        if score_buffer.dtype != torch.float32 or score_buffer.device != bank.device:
            raise ValueError("score_buffer must be CUDA float32 on bank device")
        if (
            score_buffer.shape[0] < total_sources
            or score_buffer.shape[1] != num_tokens
            or not score_buffer.is_contiguous()
        ):
            raise ValueError(
                "score_buffer must be contiguous [capacity_sources, num_tokens]"
            )
        scores = score_buffer[:total_sources]

    if output_buffer is None:
        output = torch.empty_like(bank[0])
    else:
        if output_buffer.shape != bank.shape[1:]:
            raise ValueError("output_buffer must match bank [N,D]")
        if output_buffer.dtype != bank.dtype or output_buffer.device != bank.device:
            raise ValueError("output_buffer must match bank dtype/device")
        if not output_buffer.is_contiguous():
            raise ValueError("output_buffer must be contiguous")
        output = output_buffer

    partial_tensor = partial if partial is not None else bank[0]
    block_hidden = triton.next_power_of_2(hidden_size)
    _attnres_score_kernel[(total_sources, num_tokens)](
        bank,
        partial_tensor,
        weighted_query,
        scores,
        num_tokens,
        eps=eps,
        hidden_size=hidden_size,
        bank_sources=bank_sources,
        has_partial=partial is not None,
        block_hidden=block_hidden,
        num_warps=4,
    )

    value_block_hidden = 128
    block_sources = triton.next_power_of_2(total_sources)
    _attnres_value_kernel[
        (num_tokens, triton.cdiv(hidden_size, value_block_hidden))
    ](
        bank,
        partial_tensor,
        scores,
        output,
        num_tokens,
        hidden_size=hidden_size,
        bank_sources=bank_sources,
        total_sources=total_sources,
        has_partial=partial is not None,
        block_sources=block_sources,
        block_hidden=value_block_hidden,
        num_warps=4,
    )
    return output


__all__ = ["triton_attnres_mix"]
