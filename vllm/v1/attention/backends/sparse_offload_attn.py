# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import math

import torch.torch_version

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)
from vllm.attention.backends.utils import get_flash_attn_version
from vllm.attention.ops.triton_merge_attn_states import merge_attn_states
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cdiv

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
        return [64]

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
    block_table: torch.Tensor  # TODO: logical to physical
    slot_mapping: torch.Tensor  # TODO: logical to physical

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
        if sliding_window is not None:
            raise ValueError("sliding_window is not supported")
        self.sliding_window = (-1, -1)  # TODO: check
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
                f"head_size {self.head_size} is not supported by SparseOffloadAttnBackend"
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

        # TODO: the underlying are sparse attention parameters, we fix it now
        self.topk = 4

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SparseOffloadAttentionMetadata,
        layer_name: str,
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
        key_cache, value_cache = kv_cache.unbind(0)

        # NOTE(liyi): Here we may not change the slot_mapping, cause full
        # kv_cache is stored as origin. We only select crucial kv blocks
        # for each req and rebuild a block_table for varlen_flash_attn.
        # TODO: how to get the context_manager in model_runner
        self.context_manager.update_kvcache(
            layer_name,
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        sparse_block_table, sparse_seqlens_k = varlen_sparse_kv_selection(
            query,
            key_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqlens_k=attn_metadata.seq_lens,  # seqused_k
            block_table=attn_metadata.block_table,
            topk=self.topk,
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


def varlen_sparse_kv_selection(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    # query_start_loc, used to index q, accumulated, q means query
    cu_seqlens_q: torch.Tensor,
    # seq_lens, used to index kv cache, listed, k means key
    seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    NOTE(liyi) simply modified from
        - vllm.tests.test_flash_attn.ref_paged_attn &
        - https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_naive.py#L7
    to select crucial kv blocks for each req by:
        1. estimate score by matmul(q, mean_pooling(k.T)) [Snapkv, MoBA]
        2. select top-k scores
    Here, we should calculate the gate in CPU, and then
    swap the selected kv blocks to GPU.

    return: block_table, seq_lens_k
    -------------------------------
    NOTE(yangshen)
    We are implementing a simple mean-pooling based
        block selection inside GPU now.
    - It do sparse only for decode.
    - It specify the blocks by returning a new block_table and
        cu_seqlens_k to flash_attention.

    Then we will move to InfLLM, CPU offloading and chunked prefill.
    """

    num_seqs = cu_seqlens_q.numel() - 1
    # block_table = block_table.cpu().numpy()
    new_block_table = block_table.clone()
    new_seq_lens_k = seqlens_k.clone()

    _, block_size, num_kv_heads, head_dim = key_cache.shape
    _, num_heads, _ = query.shape

    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        q_len = q_end - q_start
        k_len = seqlens_k[seq_idx].item()

        # NOTE(yangshen):
        # Since the FlashAttention only supports right align the causal masks.
        # We will not drop any blocks with query on it.
        # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#how-to-use-flashattention
        # block table:    | --------- num_sparse_blks -- | -- num_full_blks -- |
        # after selection:          | -- num_sel_blks -- | -- num_full_blks -- |
        num_seq_blks = math.ceil(k_len / block_size)
        num_full_blks = math.ceil(
            q_len / block_size
        )  # blocks for full attention
        num_sparse_blks = (
            num_seq_blks - num_full_blks
        )  # blocks for sparse attention
        if num_sparse_blks == 0 or num_sparse_blks <= topk:
            continue

        # select the blocks
        k_blks_idx = block_table[seq_idx, :num_sparse_blks]
        k_blks = key_cache[
            k_blks_idx
        ]  # [num_sparse_blocks, block_size, num_kv_heads, head_size]

        # pooling with mean
        k_blks_pooled = k_blks.mean(
            dim=1
        )  # [num_sparse_blocks, num_kv_heads, head_size]
        q = query[q_start:q_end]  # [q_len, num_heads, head_size]

        # repeat for GQA
        num_kv_groups = num_heads // num_kv_heads
        k_blks_pooled = torch.repeat_interleave(
            k_blks_pooled, num_kv_groups, dim=1
        )  # [num_sparse_blocks, num_heads, head_size]
        gate = torch.einsum(
            "qhd,khd->qhk", q, k_blks_pooled
        ).float()  # [q_len, num_heads, num_sparse_blocks]
        gate_pooled = gate.mean(dim=0).mean(dim=0)  # [num_sparse_blocks]

        _, sel_blks_idx = torch.topk(
            gate_pooled, k=topk, largest=True, sorted=False
        )  # idx in block_table

        num_sel_blks = sel_blks_idx.numel()
        new_block_table[seq_idx, :num_sel_blks] = block_table[
            seq_idx, sel_blks_idx
        ]
        new_block_table[
            seq_idx, num_sel_blks : num_sel_blks + num_full_blks
        ] = block_table[
            seq_idx, num_sparse_blks : num_sparse_blks + num_full_blks
        ]

        new_seq_lens_k[seq_idx] = (
            k_len - (num_sparse_blks - num_sel_blks) * block_size
        )

    return new_block_table, new_seq_lens_k

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
