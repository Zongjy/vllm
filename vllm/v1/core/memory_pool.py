import torch
import torch.cuda


class UnifiedMemoryCache:
    def __init__(
        self,
        cache_shape: torch.Size,
        num_cpu_blocks: dict[str, int],
        num_gpu_blocks: int,
        num_streams: int = 4,
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
        self.cpu_blocks = {}
        self.cpu_block_usage = {}
        self.cpu_lru = {}
        for layer_name, num_blocks in num_cpu_blocks.items():
            self.cpu_blocks[layer_name] = torch.zeros(
                (num_blocks, *cache_shape),
                dtype=torch.float32,
                pin_memory=True,  # Use pinned memory for faster transfers
            )
            self.cpu_block_usage[layer_name] = torch.zeros(num_blocks, dtype=torch.bool)
            self.cpu_lru[layer_name] = []

        # Initialize GPU blocks
        self.gpu_blocks = torch.zeros(
            (num_gpu_blocks, *cache_shape),
            dtype=torch.float32,
            device=self.gpu_device,
        )

        # Track block usage and status
        self.gpu_block_usage = torch.zeros(
            num_gpu_blocks, dtype=torch.bool, device=self.gpu_device
        )
        self.block_mapping = {}  # (block_id) -> (location, layer_name, block_idx)

        # LRU tracking
        self.gpu_lru = []

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

    def _get_next_stream(self):
        """Get the next available CUDA stream in a round-robin fashion."""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % self.num_streams
        return stream

    def get(self, block_id: int, target: str = "gpu", layer_name: str = None) -> torch.Tensor:
        """
        Get a block from cache. If target is 'gpu' and block is in CPU,
        it will be moved to GPU. Returns None if block not found.
        """
        if block_id not in self.block_mapping:
            self.stats["misses"] += 1
            return None

        location, block_layer, block_idx = self.block_mapping[block_id]

        # If target is GPU but block is in CPU, move it to GPU
        if target == "gpu" and location == "cpu":
            self.stats["cpu_to_gpu_fetches"] += 1
            success = self.move_to_gpu(block_id)
            if not success:
                # If move failed, just return CPU block
                if block_id not in self.pinned_blocks:
                    self.cpu_lru[block_layer].remove(block_id)
                    self.cpu_lru[block_layer].append(block_id)
                self.stats["cpu_hits"] += 1
                return self.cpu_blocks[block_layer][block_idx]
            # Get updated location after move
            location, block_layer, block_idx = self.block_mapping[block_id]

        # Update LRU and return block
        if location == "cpu":
            if block_id not in self.pinned_blocks:
                self.cpu_lru[block_layer].remove(block_id)
                self.cpu_lru[block_layer].append(block_id)
            self.stats["cpu_hits"] += 1
            return self.cpu_blocks[block_layer][block_idx]
        else:  # gpu
            if block_id not in self.pinned_blocks:
                self.gpu_lru.remove(block_id)
                self.gpu_lru.append(block_id)
            self.stats["gpu_hits"] += 1
            return self.gpu_blocks[block_idx]

    def put(
        self, block_id: int, block: torch.Tensor, target: str = "gpu", layer_name: str = None
    ) -> bool:
        """Put a block into cache. Returns True if successful."""
        if target not in ["cpu", "gpu"]:
            raise ValueError("Target must be either 'cpu' or 'gpu'")
        
        if target == "cpu":
            if layer_name is None:
                raise ValueError("layer_name must be specified for CPU target")
            if layer_name not in self.cpu_blocks:
                raise ValueError(f"Invalid layer_name: {layer_name}")
            blocks = self.cpu_blocks[layer_name]
            usage = self.cpu_block_usage[layer_name]
            lru = self.cpu_lru[layer_name]
        else:
            blocks = self.gpu_blocks
            usage = self.gpu_block_usage
            lru = self.gpu_lru

        # Find free block or evict LRU
        free_idx = torch.where(~usage)[0]
        if len(free_idx) > 0:
            idx = free_idx[0].item()
        else:
            # If target is GPU and no free blocks, try to evict to CPU first
            if target == "gpu":
                evicted = self._evict_gpu_to_cpu()
                if evicted:
                    # Try again to find a free block
                    free_idx = torch.where(~usage)[0]
                    if len(free_idx) > 0:
                        idx = free_idx[0].item()
                    else:
                        # Still no space, evict LRU
                        idx = self._evict_lru(lru, target, layer_name)
                        if idx is None:
                            return False
                else:
                    # Couldn't evict to CPU, evict LRU
                    idx = self._evict_lru(lru, target, layer_name)
                    if idx is None:
                        return False
            else:
                # For CPU target, just evict LRU
                idx = self._evict_lru(lru, target, layer_name)
                if idx is None:
                    return False

        # Store block using copy_() for pinned memory efficiency
        if target == "gpu":
            if block.device != self.gpu_device:
                # Use stream for async copy
                stream = self._get_next_stream()
                with torch.cuda.stream(stream):
                    blocks[idx].copy_(block, non_blocking=True)
                # Synchronize to ensure copy is complete
                stream.synchronize()
            else:
                blocks[idx].copy_(block)
        else:  # cpu
            if block.device != torch.device("cpu"):
                blocks[idx].copy_(block.cpu())
            else:
                blocks[idx].copy_(block)

        usage[idx] = True

        # Update mapping and LRU
        if block_id in self.block_mapping:
            # If block already exists somewhere, remove it
            old_location, old_layer, old_idx = self.block_mapping[block_id]
            if old_location == "cpu":
                self.cpu_block_usage[old_layer][old_idx] = False
                if block_id in self.cpu_lru[old_layer]:
                    self.cpu_lru[old_layer].remove(block_id)
            else:
                self.gpu_block_usage[old_idx] = False
                if block_id in self.gpu_lru:
                    self.gpu_lru.remove(block_id)

        self.block_mapping[block_id] = (target, layer_name if target == "cpu" else None, idx)
        if block_id not in self.pinned_blocks:
            lru.append(block_id)
        return True

    def _evict_lru(self, lru, target: str, layer_name: str = None):
        """
        Helper to evict LRU block. 
        Returns index of evicted block or None if no block can be evicted.
        """
        for lru_block_id in lru:
            if lru_block_id not in self.pinned_blocks:
                location, block_layer, idx = self.block_mapping[lru_block_id]
                del self.block_mapping[lru_block_id]
                lru.remove(lru_block_id)
                return idx
        return None

    def _evict_gpu_to_cpu(self):
        """Try to evict a GPU block to CPU. Returns True if successful."""
        # Try to find a non-pinned block to evict
        for block_id in self.gpu_lru:
            if block_id not in self.pinned_blocks:
                # Get block info
                _, _, gpu_idx = self.block_mapping[block_id]
                block = self.gpu_blocks[gpu_idx]

                # Try to put in each CPU layer until success
                for layer_name in self.cpu_blocks.keys():
                    if self.put(block_id, block, target="cpu", layer_name=layer_name):
                        self.stats["gpu_evictions"] += 1
                        return True
                
                # If we couldn't put it in any CPU layer, continue to next block
                continue

        return False

    def pin_block(self, block_id: int):
        """Pin a block to prevent it from being evicted."""
        if block_id in self.block_mapping:
            self.pinned_blocks.add(block_id)

    def unpin_block(self, block_id: int):
        """Unpin a block, allowing it to be evicted."""
        self.pinned_blocks.discard(block_id)

    def multi_transfer(self, block_ids: list[int], target: str = "gpu", layer_name: str = None):
        """
        Transfer multiple blocks to target location in parallel.
        For CPU target, layer_name must be specified.
        """
        if target == "cpu" and layer_name is None:
            raise ValueError("layer_name must be specified for CPU target")

        self.stats["multi_transfers"] += 1
        success = True

        # Group blocks by current location for efficient transfer
        cpu_blocks = []
        gpu_blocks = []
        for block_id in block_ids:
            if block_id in self.block_mapping:
                location, _, _ = self.block_mapping[block_id]
                if location == "cpu":
                    cpu_blocks.append(block_id)
                else:
                    gpu_blocks.append(block_id)

        # Handle transfers based on target
        if target == "gpu":
            # Move CPU blocks to GPU in parallel
            for block_id in cpu_blocks:
                if not self.move_to_gpu(block_id):
                    success = False
        else:  # target == "cpu"
            # Move GPU blocks to CPU in parallel
            for block_id in gpu_blocks:
                location, _, gpu_idx = self.block_mapping[block_id]
                block = self.gpu_blocks[gpu_idx]
                if not self.put(block_id, block, target="cpu", layer_name=layer_name):
                    success = False

        return success

    def move_to_gpu(self, block_id: int) -> bool:
        """Helper to move a CPU block to GPU. Returns True if successful."""
        if block_id not in self.block_mapping:
            return False

        location, layer_name, cpu_idx = self.block_mapping[block_id]
        if location != "cpu":
            return True  # Already in GPU

        # Try to put block in GPU
        block = self.cpu_blocks[layer_name][cpu_idx]
        return self.put(block_id, block, target="gpu")

    def get_stats(self):
        """Get cache statistics."""
        return self.stats.copy()

    def clear_stats(self):
        """Reset cache statistics."""
        for key in self.stats:
            self.stats[key] = 0
