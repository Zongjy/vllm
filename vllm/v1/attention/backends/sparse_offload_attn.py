# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import torch

import torch.torch_version

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)
from vllm.attention.backends.utils import get_flash_attn_version
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.core.scheduler_output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if current_platform.is_cuda():
    from vllm.vllm_flash_attn import flash_attn_varlen_func


logger = init_logger(__name__)


class SparseOffloadAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        # TODO: check other sizes
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SPARSE_OFFLOAD_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> Type["SparseOffloadAttentionImpl"]:
        return SparseOffloadAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return SparseOffloadAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["SparseOffloadAttentionMetadataBuilder"]:
        return SparseOffloadAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        if block_size % 16 != 0:
            raise ValueError("block_size must be a multiple of 16")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class SparseOffloadAttentionMetadata:
    # NOTE(yangshen): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    layer_name: str

    # TODO: for cascade attention

    # For logging
    num_input_tokens: int = 0  # Number of tokens including padding


class SparseOffloadAttentionMetadataBuilder:
    def __init__(self, runner: "GPUModelRunner"):
        # TODO: co processing model runner
        self.runner = runner

    def reorder_batch(
        self, input_batch: "InputBatch", scheduler_output: "SchedulerOutput"
    ):
        pass

    def build(
        self,
        num_reqs: int,
        num_actual_tokens: int,
        max_query_len: int,
        common_prefix_len: int,
    ):
        max_seq_len = self.runner.seq_lens_np[:num_reqs].max()
        query_start_loc = self.runner.query_start_loc_cpu[: num_reqs + 1].to(
            self.runner.device, non_blocking=True
        )
        seq_lens = self.runner.seq_lens_cpu[:num_reqs].to(
            self.runner.device, non_blocking=True
        )
        # TODO: two things
        block_table = self.runner.input_batch.block_table.get_device_tensor()[
            :num_reqs
        ]
        slot_mapping = (
            self.runner.slot_mapping_cpu[:num_actual_tokens]
            .to(self.runner.device, non_blocking=True)
            .long()
        )

        # we do not support the cascade attention ye
        # use_cascade = False
        # cu_prefix_query_lens = None
        # prefix_kv_lens = None
        # suffix_kv_lens = None

        attn_metadata = SparseOffloadAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            # TODO: adopt cascade attention
        )
        return attn_metadata


class SparseOffloadAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError("blocksparse_params is not supported")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads  # TODO: check
        if alibi_slopes is not None:
            raise ValueError("alibi_slopes is not supported")
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is not None:
            raise ValueError("logits_soft_cap is not supported")
        self.logits_soft_cap = 0

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = (
            SparseOffloadAttentionBackend.get_supported_head_sizes()
        )
        if self.head_size not in support_head_sizes:
            raise ValueError(
                f"head_size {self.head_size} is not supported by "
                f"SparseOffloadAttnBackend"
                f"supported head sizes are: {support_head_sizes}"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "SparseOffloadAttentonImpl"
            )

        self.vllm_flash_attn_version = get_flash_attn_version()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,  # TODO: check
        attn_metadata: SparseOffloadAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with SparseOffloadAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        """
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0, (
            "key/v_scale is not supported in SparseOffloadAttention."
        )
        assert output is not None, "output must be provided"

        if attn_metadata is None:
            # Profiling run
            return output

        # TODO: support CUDA graph

        # NOTE(yangshen): Here is the tough part - batch & transfer & sparse
        # 1. Pass the Block Manager to the forward context
        # 2. Modify the Block Manager to reallocate blocks according to query
        # 3. Correctly compute the attention scores
        num_actual_tokens = attn_metadata.num_actual_tokens

        # NOTE(liyi): Transfer full control of kvcache to context_manager
        # key_cache, value_cache = kv_cache.unbind(0)

        # TODO: how to get the context_manager in model_runner
        self.context_manager.update_kvcache(
            attn_metadata.layer_name,
            key,
            value,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        key_cache, value_cache, sparse_block_table, sparse_seqlens_k = (
            self.context_manager.load_kvcache(
                layer_name=attn_metadata.layer_name,
                query=query,
                cu_seqlens_q=attn_metadata.query_start_loc,
                seqlens_k=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                sliding_window=self.sliding_window,
                gpu_device=self.device,
            )
        )

        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=sparse_seqlens_k,  # sparse
            max_seqlen_k=attn_metadata.max_seq_len,  # TODO: update
            softmax_scale=self.scale,
            causal=True,
            window_size=(-1, -1),
            block_table=sparse_block_table,  # sparse
            softcap=self.logits_soft_cap,  # TODO check
            fa_version=self.vllm_flash_attn_version,
        )

        return output

        # (step 1) Select crucial kv blocks for each req **in CPU**

        # (step 2) Swap the selected kv blocks to GPU

        # NOTE(liyi): I'm confused about whether following things right:
        # 1. Initialize_kvcache & bind_kvcache in CPU
        # 2. In every layer's attn, send qkv to CPU
        # 3. Select crucial kv blocks in CPU
        # 4. Swap the selected kv blocks to GPU & perform flash_attn

        raise NotImplementedError(
            "SparseOffloadAttentionImpl.forward is not implemented"
        )


def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
         to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = kv.shape
    if n_rep == 1:
        return kv
    kv = kv[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return kv.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
