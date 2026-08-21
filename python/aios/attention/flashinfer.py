from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from aios.core import get_global_ctx

from .base import BaseAttentionBackend, BaseAttentionMetadata

if TYPE_CHECKING:
    from aios.core import Batch
    from aios.kvcache import MHAKVCache
    from aios.models import ModelConfig


# FlashInfer 0.5.x's paged-prefill kernels are only reliable for the prebuilt
# head dimensions below.  In particular, the legacy MiniMind 60M model uses
# head_dim=96: forced FA2 returns NaNs deterministically and the automatic
# dispatcher is input-dependent.  Keep the known fast path explicit and route
# nonstandard paged-prefill shapes through PyTorch SDPA instead.
FLASHINFER_FA2_PAGED_HEAD_DIMS = frozenset({64, 128, 256})


def select_flashinfer_paged_backend(head_dim: int) -> str:
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    return "fa2" if head_dim in FLASHINFER_FA2_PAGED_HEAD_DIMS else "auto"


def _sdpa_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cached_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """Run one causal prefill request, including a cached-prefix offset.

    Inputs use AIOS's flat NHD layout.  PyTorch SDPA expects BHLD, so this
    helper only reshapes views and repeats GQA K/V heads when needed.  A custom
    mask is required for incremental prefill because query row zero represents
    absolute position ``cached_len`` rather than position zero.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k and v must use [tokens, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[2] != k.shape[2]:
        raise ValueError("q and k must have the same head_dim")
    if q.shape[1] % k.shape[1]:
        raise ValueError("query head count must be divisible by KV head count")
    if cached_len < 0 or cached_len + q.shape[0] != k.shape[0]:
        raise ValueError("cached_len + query length must equal KV length")

    q_bhld = q.transpose(0, 1).unsqueeze(0)
    k_bhld = k.transpose(0, 1).unsqueeze(0)
    v_bhld = v.transpose(0, 1).unsqueeze(0)
    repeats = q.shape[1] // k.shape[1]
    if repeats != 1:
        k_bhld = k_bhld.repeat_interleave(repeats, dim=1)
        v_bhld = v_bhld.repeat_interleave(repeats, dim=1)

    if cached_len == 0:
        attention_mask = None
        is_causal = True
    else:
        query_positions = torch.arange(
            cached_len,
            cached_len + q.shape[0],
            device=q.device,
        )
        key_positions = torch.arange(k.shape[0], device=q.device)
        attention_mask = (
            key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        ).unsqueeze(0).unsqueeze(0)
        is_causal = False

    output = F.scaled_dot_product_attention(
        q_bhld,
        k_bhld,
        v_bhld,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=sm_scale,
    )
    return output.squeeze(0).transpose(0, 1).contiguous()


@dataclass
class FlashInferAttentionMetadata(BaseAttentionMetadata):
    cu_seqlens_q_cpu: torch.Tensor
    cu_seqlens_k_cpu: torch.Tensor
    cu_seqlens_q_gpu: torch.Tensor
    indices: torch.Tensor
    last_page_len_cpu: torch.Tensor
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    pos_encoding_mode: str
    seq_lens_cpu: torch.Tensor
    dtype: torch.dtype
    wrapper: Any
    initialized: bool = False

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs].to(torch.long) - 1


class FlashInferAttentionBackend(BaseAttentionBackend):
    """FlashInfer paged-attention backend for AIOS.

    This mirrors mini-sglang's backend split while keeping metadata creation
    local to the backend for now. Both prefill and decode use flat q/k/v
    tensors: (total_tokens, heads, head_dim).
    """

    def __init__(
        self,
        config: ModelConfig,
        workspace_size: int = 128 * 1024 * 1024,
    ) -> None:
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.sm_scale = config.head_dim ** -0.5
        self.prefill_kernel_backend = (
            "fa2"
            if config.head_dim in FLASHINFER_FA2_PAGED_HEAD_DIMS
            else "torch-sdpa"
        )
        self.decode_kernel_backend = select_flashinfer_paged_backend(
            config.head_dim
        )
        self.backend_name = (
            "flashinfer-paged-"
            f"prefill-{self.prefill_kernel_backend}-"
            f"decode-{self.decode_kernel_backend}"
        )
        self.workspace_size = workspace_size
        self._workspace: torch.Tensor | None = None
        self._prefill_wrapper: Any | None = None
        self._decode_wrapper: Any | None = None
        self._cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_kv_cache: MHAKVCache,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        if batch.is_prefill:
            return self._prefill(q, k, v, paged_kv_cache, layer_id, batch)
        return self._decode(q, k, v, paged_kv_cache, layer_id, batch)

    def _get_prefill_wrapper(self, device: torch.device):
        assert device.type == "cuda", "FlashInfer attention requires CUDA"
        if self._prefill_wrapper is None:
            import flashinfer

            if self._workspace is None:
                self._workspace = torch.empty(
                    self.workspace_size, dtype=torch.uint8, device=device
                )
            if self.prefill_kernel_backend == "torch-sdpa":
                raise RuntimeError("torch-sdpa prefill does not use a FlashInfer wrapper")
            self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self._workspace,
                kv_layout="NHD",
                backend=self.prefill_kernel_backend,
            )
        return self._prefill_wrapper

    def _get_decode_wrapper(self, device: torch.device):
        assert device.type == "cuda", "FlashInfer attention requires CUDA"
        if self._decode_wrapper is None:
            import flashinfer

            if self._workspace is None:
                self._workspace = torch.empty(
                    self.workspace_size, dtype=torch.uint8, device=device
                )
            use_tensor_cores = self.num_heads // self.num_kv_heads >= 4
            self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._workspace,
                use_tensor_cores=use_tensor_cores,
                kv_layout="NHD",
                backend=self.decode_kernel_backend,
            )
        return self._decode_wrapper

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self._cached_ones_cpu):
            return self._cached_ones_cpu[:bs]
        next_len = 1 << (bs - 1).bit_length()
        self._cached_ones_cpu = torch.ones(
            next_len, dtype=torch.int32, pin_memory=True
        )
        return self._cached_ones_cpu[:bs]

    def prepare_metadata(self, batch: Batch) -> FlashInferAttentionMetadata:
        ctx = get_global_ctx()
        page_table = ctx.page_table
        assert page_table.is_cuda, "FlashInfer attention requires a CUDA page table"
        reqs = batch.reqs
        batch_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        device = page_table.device

        seq_lens_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)

        metadata = FlashInferAttentionMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat(
                [page_table[req.table_idx, : req.device_len] for req in reqs]
            ),
            last_page_len_cpu=self._get_ones_cpu(batch_size),
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_lens_cpu,
            dtype=ctx.kv_cache.dtype,
            wrapper=(
                self._get_decode_wrapper(device)
                if batch.is_decode
                else (
                    None
                    if self.prefill_kernel_backend == "torch-sdpa"
                    else self._get_prefill_wrapper(device)
                )
            ),
        )
        batch.attn_metadata = metadata
        return metadata

    def _initialize_metadata_once(self, metadata: FlashInferAttentionMetadata) -> None:
        if metadata.initialized:
            return

        metadata.initialized = True
        if metadata.wrapper is self._decode_wrapper:
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )

    def _kv_cache_for_flashinfer(
        self, paged_kv_cache: MHAKVCache, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_cache = paged_kv_cache.k_cache(layer_id)
        v_cache = paged_kv_cache.v_cache(layer_id)
        k_cache = k_cache.view(-1, 1, k_cache.shape[2], k_cache.shape[3])
        v_cache = v_cache.view(-1, 1, v_cache.shape[2], v_cache.shape[3])
        return k_cache, v_cache

    def _prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_kv_cache: MHAKVCache,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        assert batch.is_prefill, "prefill backend expects a prefill batch"
        metadata = batch.attn_metadata
        assert isinstance(metadata, FlashInferAttentionMetadata)
        paged_kv_cache.store_kv(k, v, batch.out_loc.view(-1), layer_id)
        if self.prefill_kernel_backend == "torch-sdpa":
            return self._prefill_with_sdpa(
                q,
                k,
                v,
                paged_kv_cache,
                layer_id,
                batch,
            )
        self._initialize_metadata_once(metadata)
        attn_output = metadata.wrapper.run(
            q, self._kv_cache_for_flashinfer(paged_kv_cache, layer_id)
        )
        return attn_output.reshape(q.size(0), -1)

    def _prefill_with_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_kv_cache: MHAKVCache,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        """Correctness fallback for unsupported FlashInfer paged head dims."""
        ctx = get_global_ctx()
        k_cache = paged_kv_cache.k_cache(layer_id).view(
            -1, self.num_kv_heads, self.head_dim
        )
        v_cache = paged_kv_cache.v_cache(layer_id).view(
            -1, self.num_kv_heads, self.head_dim
        )
        outputs: list[torch.Tensor] = []
        extend_offset = 0
        for req in batch.reqs:
            extend_len = req.extend_len
            next_offset = extend_offset + extend_len
            query = q[extend_offset:next_offset]
            if req.cached_len == 0:
                keys = k[extend_offset:next_offset]
                values = v[extend_offset:next_offset]
            else:
                page_indices = ctx.page_table[
                    req.table_idx, : req.device_len
                ].to(torch.long)
                keys = k_cache[page_indices]
                values = v_cache[page_indices]
            outputs.append(
                _sdpa_prefill_attention(
                    query,
                    keys,
                    values,
                    cached_len=req.cached_len,
                    sm_scale=self.sm_scale,
                )
            )
            extend_offset = next_offset
        if extend_offset != q.shape[0]:
            raise RuntimeError("prefill request lengths do not match flat Q tokens")
        return torch.cat(outputs, dim=0).reshape(q.size(0), -1)

    def _decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_kv_cache: MHAKVCache,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        assert batch.is_decode, "decode backend expects a decode batch"
        bsz = batch.size
        assert q.size(0) == bsz, "decode batch must contain one flat token per request"
        metadata = batch.attn_metadata
        assert isinstance(metadata, FlashInferAttentionMetadata)
        self._initialize_metadata_once(metadata)
        paged_kv_cache.store_kv(k, v, batch.out_loc.view(-1), layer_id)
        attn_output = metadata.wrapper.run(
            q,
            self._kv_cache_for_flashinfer(paged_kv_cache, layer_id),
        )
        return attn_output.reshape(bsz, -1)
