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
    def __init__(self, vllm_config: VllmConfig, device: torch.device="cuda"):
        self.vllm_config = vllm_config
        self.device = device
        # NOTE(liyi): Lazy initialization after model_runner loads the model
        self.kv_cache = None
        # TODO: the underlying are sparse attention parameters, we fix it now
        self.topk = 4
        # Track current positions in blocks
        self.block_positions = defaultdict(dict)  # layer_name -> block_id -> current_pos

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
            Update the kv cache with the given slot mapping.
            slot_mapping maps token_id to block_id, 
            e.g.:
                [0, 0, K, K, K+1, K+1, K+2, 2K, 2K, 2K+1] 
            where K is max_num_blocks_per_req
            and block size is 2.
        """
        if layer_name not in self.block_positions:
            self.block_positions[layer_name] = defaultdict(int)
        
        # Get block size from cache shape
        block_size = self.kv_cache.cache_shape[0]
        
        # For each token, store its key/value in the corresponding block
        for token_idx, block_id in enumerate(slot_mapping):
            block_id = block_id.item()
            current_pos = self.block_positions[layer_name][block_id]
            
            # Create a new block on GPU if needed
            gpu_idx = self.kv_cache.get(block_id, target="gpu")
            if gpu_idx is None:
                # Initialize a new block
                new_block = torch.zeros(
                    self.kv_cache.cache_shape,
                    dtype=key.dtype,
                    device=self.device
                )
                # Put the new block in GPU cache
                self.kv_cache.put(block_id, new_block, target="gpu")
                gpu_idx = self.kv_cache.get(block_id, target="gpu")
                if gpu_idx is None:
                    raise RuntimeError(f"Failed to allocate GPU block for block_id {block_id}")
            
            # Copy key/value to the block at current position
            self.kv_cache.gpu_blocks[gpu_idx, current_pos].copy_(key[token_idx])
            self.kv_cache.gpu_blocks[gpu_idx, current_pos].copy_(value[token_idx])
            
            # Update position, ensuring we don't exceed block size
            current_pos = (current_pos + 1) % block_size
            self.block_positions[layer_name][block_id] = current_pos

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
        Returns: (gpu_block_table, gpu_key_cache, gpu_value_cache, sparse_seqlens_k)
        """
        # Get all CPU blocks for this layer
        cpu_blocks = self.kv_cache.get_cpu_blocks(layer_name)

        # Perform sparse selection on CPU
        sparse_block_table, sparse_seqlens_k = varlen_sparse_kv_selection(
            query.to("cpu"),
            cpu_blocks.unbind(0),
            cu_seqlens_q=cu_seqlens_q.to("cpu"),
            seqlens_k=seqlens_k.to("cpu"),
            block_table=block_table.to("cpu"),
            topk=self.topk,
            sliding_window=sliding_window,
        )

        # Move selected blocks to GPU and build GPU block table
        gpu_block_mapping = {}  # original_block_id -> gpu_block_id
        for block_id in sparse_block_table.unique():
            gpu_idx = self.kv_cache.get(block_id.item(), target="gpu")
            if gpu_idx is None:
                raise ValueError(f"Failed to move block {block_id} to GPU")
            gpu_block_mapping[block_id.item()] = gpu_idx

        # Create GPU block table using the mapping
        gpu_block_table = torch.tensor(
            [gpu_block_mapping[bid.item()] for bid in sparse_block_table.flatten()],
            device=gpu_device
        ).view_as(sparse_block_table)

        # Split GPU blocks into separate key and value caches
        gpu_blocks = self.kv_cache.gpu_blocks
        gpu_key_cache, gpu_value_cache = gpu_blocks.unbind(0)

        return gpu_block_table, gpu_key_cache, gpu_value_cache, sparse_seqlens_k


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
