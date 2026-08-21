from __future__ import annotations

import pytest
import torch

from aios.attention import select_flashinfer_paged_backend
from aios.attention.flashinfer import _sdpa_prefill_attention


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_known_fa2_paged_head_dims_keep_explicit_fast_path(head_dim: int) -> None:
    assert select_flashinfer_paged_backend(head_dim) == "fa2"


@pytest.mark.parametrize("head_dim", [80, 96, 192])
def test_nonstandard_head_dims_use_safe_flashinfer_dispatch(head_dim: int) -> None:
    assert select_flashinfer_paged_backend(head_dim) == "auto"


def test_flashinfer_backend_selector_rejects_invalid_head_dim() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_flashinfer_paged_backend(0)


@pytest.mark.parametrize("cached_len", [0, 2])
def test_sdpa_prefill_fallback_matches_explicit_causal_attention(
    cached_len: int,
) -> None:
    generator = torch.Generator().manual_seed(17)
    query_len = 3
    kv_len = cached_len + query_len
    q = torch.randn(query_len, 4, 8, generator=generator)
    k = torch.randn(kv_len, 2, 8, generator=generator)
    v = torch.randn(kv_len, 2, 8, generator=generator)
    scale = 8**-0.5

    actual = _sdpa_prefill_attention(
        q,
        k,
        v,
        cached_len=cached_len,
        sm_scale=scale,
    )
    repeated_k = k.repeat_interleave(2, dim=1)
    repeated_v = v.repeat_interleave(2, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, repeated_k) * scale
    query_positions = torch.arange(cached_len, cached_len + query_len)
    key_positions = torch.arange(kv_len)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    weights = scores.softmax(dim=-1)
    expected = torch.einsum("hqk,khd->qhd", weights, repeated_v)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
