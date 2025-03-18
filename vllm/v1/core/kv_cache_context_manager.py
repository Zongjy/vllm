import torch

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
)
from vllm.v1.attention.backends.sparse_offload_attn import SparseOffloadAttentionBackend
from vllm.model_executor.models.utils import extract_layer_index
from collections import defaultdict
from vllm.utils import cdiv
from vllm.v1.core.memory_pool import UnifiedMemoryCache
from typing import Optional, Dict, List, Tuple


class KVCacheContextManager:
    # TODO(liyi): Unified with KVCacheManager for:
    # 1. token_budget calculation
    # 2. LRU policy for swapping kvcache blocks
    def __init__(self, vllm_config: VllmConfig, device: torch.device="cuda"):
        self.vllm_config = vllm_config
        self.device = device
        # NOTE(liyi): Lazy initialization after model_runner loads the model
        self.kv_cache = None
        # TODO: the underlying are sparse attention parameters, we fix it now
        self.topk = 4
        # Track the current block and position for each layer

    def _initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self.layer2index = {}
        if len(kv_cache_config.groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        
        # Get the first layer's spec to determine shared parameters
        first_layer_spec = kv_cache_config[
            list(kv_cache_config.kv_cache_spec.keys())[0]]
        kv_cache_shape = (
            first_layer_spec.block_size,
            first_layer_spec.num_kv_heads,
            first_layer_spec.head_size,
        )
        dtype = first_layer_spec.dtype

        # Calculate num_blocks for each layer
        num_cpu_blocks = {}
        for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
            tensor_config = kv_cache_config.tensors[layer_name]
            assert tensor_config.size % layer_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // layer_spec.page_size_bytes

            # Verify all layers have same shape
            assert isinstance(layer_spec, FullAttentionSpec)
            assert layer_spec.dtype == first_layer_spec.dtype
            assert layer_spec.block_size == first_layer_spec.block_size
            assert layer_spec.num_kv_heads == first_layer_spec.num_kv_heads
            assert layer_spec.head_size == first_layer_spec.head_size
            
            num_cpu_blocks[layer_name] = num_blocks
            self.current_block_info[layer_name] = (0, 0)  # (block_id, position)

        # Initialize UnifiedMemoryCache with layer-specific CPU blocks
        total_blocks = sum(num_cpu_blocks.values())
        self.kv_cache = UnifiedMemoryCache(
            cache_shape=kv_cache_shape,
            num_cpu_blocks=num_cpu_blocks,
            num_gpu_blocks=total_blocks // 2,
            num_streams=4,
        )

    def update_kvcache(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """
        Update the kv cache with the given slot mapping & block_table.
        The key and value are for a single token, and need to be stored in the current block.
        """
        if layer_name not in self.current_block_info:
            raise ValueError(f"Layer {layer_name} not initialized")

        raise NotImplementedError

    def load_kvcache(
        self,
        layer_name: str,
        query: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqlens_k: torch.Tensor,
        block_table: torch.Tensor,
        sliding_window: Optional[int],
        gpu_device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load kv cache for the given layer with sparse selection.
        Returns: (gpu_block_table, kvcache_k, kvcache_v, sparse_seqlens_k)
        """
        # NOTE(liyi): need fix
        # First get the blocks from CPU
        cpu_blocks = []
        for block_id in block_table.unique():
            block = self.kv_cache.get(block_id, target="cpu", layer_name=layer_name)
            if block is not None:
                cpu_blocks.append(block)

        if not cpu_blocks:
            raise ValueError(f"No blocks found for layer {layer_name}")

        # Stack blocks into a single tensor for sparse selection
        key_cache = torch.stack(cpu_blocks, dim=0)

        # Perform sparse selection
        sparse_block_table, sparse_seqlens_k = varlen_sparse_kv_selection(
            query.to(self.device),
            key_cache,
            cu_seqlens_q=cu_seqlens_q.to(self.device),
            seqlens_k=seqlens_k.to(self.device),
            block_table=block_table.to(self.device),
            topk=self.topk,
            sliding_window=sliding_window,
        )

        # Transfer selected blocks to GPU
        gpu_blocks = []
        gpu_block_mapping = {}  # old_block_id -> new_gpu_block_id
        for i, block_id in enumerate(sparse_block_table.unique()):
            block = self.kv_cache.get(block_id, target="cpu", layer_name=layer_name)
            if block is None:
                raise ValueError(f"Block {block_id} not found in layer {layer_name}")
            
            # Move block to GPU
            gpu_block = block.to(gpu_device)
            self.kv_cache.put(i, gpu_block, target="gpu")
            gpu_blocks.append(gpu_block)
            gpu_block_mapping[block_id.item()] = i

        # Create new GPU block table
        gpu_block_table = torch.tensor([gpu_block_mapping[bid.item()] 
                                      for bid in sparse_block_table.flatten()],
                                     device=gpu_device).view_as(sparse_block_table)

        # Stack GPU blocks into key and value caches
        kvcache = torch.stack(gpu_blocks, dim=0)  # [num_blocks, block_size, num_heads, head_size]
        kvcache_k, kvcache_v = kvcache.unbind(0)  # Split into key and value caches

        return gpu_block_table, kvcache_k, kvcache_v, sparse_seqlens_k


def varlen_sparse_kv_selection(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    # query_start_loc, used to index q, accumulated, q means query
    cu_seqlens_q: torch.Tensor,
    # seq_lens, used to index kv cache, listed, k means key
    seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
    sliding_window: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Simply modified from
            - vllm.tests.test_flash_attn.ref_paged_attn &
            - https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_naive.py#L7
        to select crucial kv blocks for each req by:
            1. estimate score by matmul(q, mean_pooling(k.T)) [Snapkv, MoBA]
            2. select top-k scores
        And 
            - It do sparse only for decode.
            - It specify the blocks by returning a new block_table and
                cu_seqlens_k to flash_attention.

        return: block_table, seq_lens_k
    """

    num_seqs = cu_seqlens_q.numel() - 1
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
        # Since the FlashAttention only supports right align the
        # causal masks. We will not drop any blocks with query on it.
        # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#how-to-use-flashattention
        # NOTE(liyi): Apply sliding_window here
        # block table:
        # before: | < sink > |     < sparse >     |  < max(local, full) >  |
        # after:  | < sink > |  < select > |  < max(local, full) > |
        num_sink_blks, num_local_blks = (
            sliding_window if sliding_window else (0, 0)
        )
        num_seq_blks = cdiv(k_len, block_size)
        num_local_blks = max(
            cdiv(q_len, block_size),
            num_local_blks,
        )  # blocks for local attention
        num_sparse_blks = (
            num_seq_blks - num_sink_blks - num_local_blks
        )  # blocks for sparse attention
        if num_sparse_blks == 0 or num_sparse_blks <= topk:
            continue

        # select the blocks
        k_blks_idx = block_table[
            seq_idx, num_sink_blks : num_sink_blks + num_sparse_blks
        ]
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
        new_block_table[seq_idx, :num_sink_blks] = block_table[
            seq_idx, :num_sink_blks
        ]  # sink blocks

        new_block_table[
            seq_idx, num_sink_blks : num_sink_blks + num_sel_blks
        ] = block_table[seq_idx, sel_blks_idx]  # selected blocks

        new_block_table[
            seq_idx,
            num_sink_blks + num_sel_blks : num_sink_blks
            + num_sel_blks
            + num_local_blks,
        ] = block_table[
            seq_idx,
            num_sink_blks + num_sparse_blks : num_sink_blks
            + num_sparse_blks
            + num_local_blks,
        ]  # local block

        new_seq_lens_k[seq_idx] = (
            k_len - (num_sparse_blks - num_sel_blks) * block_size
        )

    return new_block_table, new_seq_lens_k
