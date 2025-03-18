import torch
import torch.cuda
from typing import Optional


class UnifiedMemoryCache:
    gpu_device: torch.device
    cache_shape: torch.Size
    # each layer have number of cpu blocks
    layer_names: list[str]
    num_cpu_blocks: int
    num_gpu_blocks: int

    # use multiple cuda stream to accelerate memory transfer
    num_streams: int
    streams: list[torch.cuda.Stream]
    current_stream: int

    # lru related
    cpu_blocks: dict[str, torch.Tensor]
    cpu_block_usage: dict[str, torch.Tensor]
    gpu_blocks: torch.Tensor
    gpu_block_usage: torch.Tensor
    gpu_lru: list[int]
    pinned_blocks: set[int]

    # (layer_name, block_id) -> (location, block_idx)
    # the block in the CPU, the block_idx is the index in the CPU block
    # and also is the block_id
    layer_block_mapping: dict[tuple[str, int], tuple[str, int]]
    # (gpu_block_idx) -> (layer_name, block_id)
    gpu_block_mapping: dict[int, tuple[str, int]]
    stats: dict[str, int]

    def __init__(
        self,
        # KV Cache shape
        cache_shape: torch.Size,
        # the cpu block for each layer like the a form
        # of {layer_name: num_blocks}
        layer_names: list[str],
        num_cpu_blocks: int,
        num_gpu_blocks: int,
        num_streams: int = 4,
        kv_cache_dtype: torch.dtype = torch.float32,
    ):
        self.gpu_device = torch.device("cuda")
        self.cache_shape = cache_shape
        self.num_cpu_blocks = num_cpu_blocks
        self.num_gpu_blocks = num_gpu_blocks

        # Create CUDA streams for parallel transfers
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0

        # Initialize CPU blocks with pinned memory for each layer
        self.cpu_blocks = dict()
        self.cpu_block_usage = dict()

        for layer_name in layer_names:
            self.cpu_blocks[layer_name] = torch.zeros(
                (num_cpu_blocks, *cache_shape),
                dtype=kv_cache_dtype,
                # Use pinned memory for faster transfers
                pin_memory=True,
            )
            self.cpu_block_usage[layer_name] = torch.zeros(
                num_cpu_blocks, dtype=torch.bool
            )

        # Initialize GPU blocks
        # [num_blocks, block_size, num_heads, head_size, 2]
        self.gpu_blocks = torch.zeros(
            (num_gpu_blocks, *cache_shape),
            dtype=kv_cache_dtype,
            device=self.gpu_device,
        )

        # Track block usage and status
        self.gpu_block_usage = torch.zeros(
            num_gpu_blocks, dtype=torch.bool, device=self.gpu_device
        )

        # (block_id) -> (location, layer_name, block_idx)
        self.layer_block_mapping = dict()

        # LRU tracking
        self.gpu_lru = list()

        # Pinned blocks tracking
        self.pinned_blocks = set()

        # Stats tracking
        self.stats = {
            "gpu_evictions": 0,
            "cpu_to_gpu_fetches": 0,
            "gpu_hits": 0,
            "cpu_hits": 0,
            "misses": 0,
            "multi_transfers": 0,
        }
        self.cache_shape = cache_shape
    
    def get_cache_shape(self):
        return self.cache_shape

    def _get_next_stream(self):
        """Get the next available CUDA stream in a round-robin fashion."""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % self.num_streams
        return stream

    def get_layer_cpu_blocks(self, layer_name: str) -> torch.Tensor:
        return self.cpu_blocks[layer_name]

    def get_gpu_blocks(self) -> torch.Tensor:
        """
        get all gpu blocks, for flash_attention only
        """
        return self.gpu_blocks

    def get(self, layer_name: str, block_id: int, target: str = "gpu") -> int:
        """
        Get a block from cache. If target is 'gpu' and block is in CPU,
        it will be moved to GPU. Returns the block index in target location,
        or None if block not found.
        """
        if (layer_name, block_id) not in self.layer_block_mapping:
            raise RuntimeError("Block not found in cache")

        location, block_idx = self.layer_block_mapping[(layer_name, block_id)]

        # If target is GPU but block is in CPU, move it to GPU
        if target == "gpu" and location == "cpu":
            self.stats["cpu_to_gpu_fetches"] += 1
            success = self._move_to_gpu(layer_name, block_id)
            if not success:
                raise RuntimeError(
                    "The gpu memory usage is full, please add more memory"
                )
            # Get updated location after move
            location, block_idx = self.layer_block_mapping[
                (layer_name, block_id)
            ]
        elif location == "gpu":
            self.stats["gpu_hits"] += 1

        if block_idx not in self.pinned_blocks:
            self.gpu_lru.remove(block_idx)
            self.gpu_lru.append(block_idx)

        return block_idx

    def put(
        self,
        layer_name: str,
        block_id: int,
        block: torch.Tensor,
        target: str = "gpu",
    ) -> None:
        """Put a block into cache. This function will be success or die."""
        if target not in ["cpu", "gpu"]:
            raise ValueError("Target must be either 'cpu' or 'gpu'")

        if target == "cpu":
            if layer_name not in self.cpu_blocks:
                raise ValueError(f"Invalid layer_name: {layer_name}")
            # for cpu, just copy it to the block
            self.cpu_blocks[layer_name][block_id].copy_(block)
            self.layer_block_mapping[(layer_name, block_id)] = ("cpu", block_id)
            return True
        else:
            blocks = self.gpu_blocks
            usage = self.gpu_block_usage

        # Find free block or evict LRU
        free_idx = torch.where(~usage)[0]
        if len(free_idx) > 0:
            idx = free_idx[0].item()
        else:
            # If target is GPU and no free blocks, try to evict to CPU first
            evicted = self._evict_gpu_to_cpu()
            if evicted is not None:
                # Try again to find a free block
                idx = evicted
            else:
                raise RuntimeError(
                    "The gpu memory usage is full, please add more memory"
                )

        # Store block using copy_() for pinned memory efficiency
        blocks[idx].copy_(block)

        self.gpu_block_usage[idx] = True

        # Update mapping and LRU
        if (layer_name, block_id) in self.layer_block_mapping:
            # If block already exists somewhere, remove it
            old_location, old_idx = self.layer_block_mapping[
                (layer_name, block_id)
            ]
            if old_location == "cpu":
                self.cpu_block_usage[layer_name][old_idx] = False
            else:
                self.gpu_block_usage[old_idx] = False
                if block_id in self.gpu_lru:
                    self.gpu_lru.remove(block_id)

        self.layer_block_mapping[layer_name][block_id] = (
            target,
            idx,
        )
        self.gpu_lru.append(block_id)
        return True

    # Try to evict a GPU block to CPU
    def _evict_gpu_to_cpu(self) -> Optional[int]:
        """Try to evict a GPU block to CPU. Returns block if successful."""
        # Try to find a non-pinned block to evict
        for block_idx in self.gpu_lru:
            if block_idx not in self.pinned_blocks:
                # Get block info
                layer_name, block_id = self.gpu_block_mapping[block_idx]
                block = self.gpu_blocks[block_idx]

                self.put(block_id, layer_name, block, target="cpu")

                self.gpu_block_usage[block_idx] = False
                self.gpu_lru.remove(block_idx)
                return block_idx
        # if no non-pinned block found, return None
        return None

    def pin_block(self, layer_name: str, block_id: int):
        """Pin a block to prevent it from being evicted."""
        if (layer_name, block_id) in self.layer_block_mapping:
            location, idx = self.layer_block_mapping[(layer_name, block_id)]
            if location != "gpu":
                raise RuntimeError("Only GPU blocks can be pinned")
            self.pinned_blocks.add(idx)
        else:
            raise RuntimeError("Block not found in cache")

    def unpin_block(self, block_id: int):
        """Unpin a block, allowing it to be evicted."""
        self.pinned_blocks.discard(block_id)

    def batch_get(self, layer_name: str, block_ids: list[int]) -> list[int]:
        """
        Transfer multiple blocks to target location in parallel.
        For CPU target, layer_name must be specified.
        """
        # Group blocks by current location for efficient transfer
        cpu_blocks = []
        for block_id in block_ids:
            location, idx = self.layer_block_mapping[(layer_name, block_id)]
            if location == "cpu":
                cpu_blocks.append(idx)

        # Handle transfers based on target
        free_idx = torch.where(~self.gpu_block_usage)[0].tolist()
        if len(free_idx) < len(cpu_blocks):
            # should evict from GPU
            needed_blocks = len(cpu_blocks) - len(free_idx)
            for _ in range(needed_blocks):
                free_idx.append(self._evict_gpu_to_cpu())

        for cpu_block in cpu_blocks:
            streams = [self._get_next_stream() for _ in range(len(cpu_blocks))]
            for cpu_block, gpu_free_idx, stream in zip(
                cpu_blocks, free_idx, streams
            ):
                with torch.cuda.stream(stream):
                    self.gpu_blocks[free_idx].copy_(
                        self.cpu_blocks[layer_name][cpu_block],
                        non_blocking=True,
                    )
                    self.layer_block_mapping[(layer_name, cpu_block)] = (
                        "gpu",
                        free_idx,
                    )
                # do lru
                self.gpu_lru.remove(free_idx)
                self.gpu_lru.append(free_idx)

        for stream in self.streams:
            stream.synchronize()
        return free_idx

    def _move_to_gpu(self, layer_name: str, block_id: int) -> bool:
        """Helper to move a CPU block to GPU. Returns True if successful."""
        if (layer_name, block_id) not in self.layer_block_mapping:
            raise RuntimeError("Block not found in cache")

        location, cpu_idx = self.layer_block_mapping[(layer_name, block_id)]
        if location != "cpu":
            return True  # Already in GPU

        # Try to put block in GPU
        block = self.cpu_blocks[layer_name][cpu_idx]
        return self.put(block_id, block, target="gpu", layer_name=layer_name)

    def get_stats(self):
        """Get cache statistics."""
        return self.stats.copy()

    def clear_stats(self):
        """Reset cache statistics."""
        for key in self.stats:
            self.stats[key] = 0
