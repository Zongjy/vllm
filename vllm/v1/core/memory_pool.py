import torch
import torch.cuda


class UnifiedMemoryCache:
    def __init__(
        self,
        cache_shape: torch.Size,
        num_cpu_blocks: int,
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

        # Initialize CPU blocks with pinned memory
        self.cpu_blocks = torch.zeros(
            (num_cpu_blocks, *cache_shape),
            dtype=torch.float32,
            pin_memory=True,  # Use pinned memory for faster transfers
        )

        # Initialize GPU blocks
        self.gpu_blocks = torch.zeros(
            (num_gpu_blocks, *cache_shape),
            dtype=torch.float32,
            device=self.gpu_device,
        )

        # Track block usage and status
        self.cpu_block_usage = torch.zeros(num_cpu_blocks, dtype=torch.bool)
        self.gpu_block_usage = torch.zeros(
            num_gpu_blocks, dtype=torch.bool, device=self.gpu_device
        )
        self.block_mapping = {}

        # LRU tracking
        self.cpu_lru = []
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

    def get(self, block_id: int, target: str = "gpu") -> torch.Tensor:
        """
        Get a block from cache. If target is 'gpu' and block is in CPU,
        it will be moved to GPU. Returns None if block not found.
        """
        if block_id not in self.block_mapping:
            self.stats["misses"] += 1
            return None

        location, block_idx = self.block_mapping[block_id]

        # If target is GPU but block is in CPU, move it to GPU
        if target == "gpu" and location == "cpu":
            self.stats["cpu_to_gpu_fetches"] += 1
            success = self.move_to_gpu(block_id)
            if not success:
                # If move failed, just return CPU block
                if block_id not in self.pinned_blocks:
                    self.cpu_lru.remove(block_id)
                    self.cpu_lru.append(block_id)
                self.stats["cpu_hits"] += 1
                return self.cpu_blocks[block_idx]
            # Get updated location after move
            location, block_idx = self.block_mapping[block_id]

        # Update LRU and return block
        if location == "cpu":
            if block_id not in self.pinned_blocks:
                self.cpu_lru.remove(block_id)
                self.cpu_lru.append(block_id)
            self.stats["cpu_hits"] += 1
            return self.cpu_blocks[block_idx]
        else:  # gpu
            if block_id not in self.pinned_blocks:
                self.gpu_lru.remove(block_id)
                self.gpu_lru.append(block_id)
            self.stats["gpu_hits"] += 1
            return self.gpu_blocks[block_idx]

    def put(
        self, block_id: int, block: torch.Tensor, target: str = "gpu"
    ) -> bool:
        """Put a block into cache. Returns True if successful."""
        if target not in ["cpu", "gpu"]:
            raise ValueError("Target must be either 'cpu' or 'gpu'")

        if target == "cpu":
            blocks = self.cpu_blocks
            usage = self.cpu_block_usage
            lru = self.cpu_lru
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
                        idx = self._evict_lru(lru)
                        if idx is None:
                            return False
                else:
                    # Couldn't evict to CPU, evict LRU
                    idx = self._evict_lru(lru)
                    if idx is None:
                        return False
            else:
                # For CPU target, just evict LRU
                idx = self._evict_lru(lru)
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
            old_location, old_idx = self.block_mapping[block_id]
            if old_location == "cpu":
                self.cpu_block_usage[old_idx] = False
                if block_id in self.cpu_lru:
                    self.cpu_lru.remove(block_id)
            else:
                self.gpu_block_usage[old_idx] = False
                if block_id in self.gpu_lru:
                    self.gpu_lru.remove(block_id)

        self.block_mapping[block_id] = (target, idx)
        if block_id not in self.pinned_blocks:
            lru.append(block_id)
        return True

    def _evict_lru(self, lru):
        """
            Helper to evict LRU block. 
            Returns index of evicted block or None if no block can be evicted.
        """
        for lru_block_id in lru:
            if lru_block_id not in self.pinned_blocks:
                location, idx = self.block_mapping[lru_block_id]
                del self.block_mapping[lru_block_id]
                lru.remove(lru_block_id)
                return idx
        return None

    def _evict_gpu_to_cpu(self):
        """
            Evict least recently used GPU block to CPU.
            Returns True if successful, False otherwise.
        """
        # Check if CPU has space
        cpu_free_idx = torch.where(~self.cpu_block_usage)[0]
        if len(cpu_free_idx) == 0:
            # Try to evict CPU LRU to make space
            cpu_idx = self._evict_lru(self.cpu_lru)
            if cpu_idx is None:
                return False
            cpu_free_idx = torch.tensor([cpu_idx])

        # Find GPU LRU block to evict
        for block_id in self.gpu_lru:
            if block_id not in self.pinned_blocks:
                # Get the block
                _, gpu_idx = self.block_mapping[block_id]
                block = self.gpu_blocks[gpu_idx]

                # Move to CPU using pinned memory copy
                cpu_idx = cpu_free_idx[0].item()
                # Use stream for async copy
                stream = self._get_next_stream()
                with torch.cuda.stream(stream):
                    self.cpu_blocks[cpu_idx].copy_(block, non_blocking=True)
                # Synchronize to ensure copy is complete
                stream.synchronize()

                self.cpu_block_usage[cpu_idx] = True

                # Update mappings
                self.gpu_block_usage[gpu_idx] = False
                self.gpu_lru.remove(block_id)
                self.block_mapping[block_id] = ("cpu", cpu_idx)
                self.cpu_lru.append(block_id)

                self.stats["gpu_evictions"] += 1
                return True

        return False

    def pin(self, block_id: int) -> bool:
        """
            Pin a block in cache to prevent eviction. 
            Returns True if block exists.
        """
        if block_id in self.block_mapping:
            self.pinned_blocks.add(block_id)
            return True
        return False

    def unpin(self, block_id: int) -> bool:
        """
            Unpin a block in cache. 
            Returns True if block was pinned.
        """
        if block_id in self.pinned_blocks:
            self.pinned_blocks.remove(block_id)
            return True
        return False

    def move_to_gpu(self, block_id: int) -> bool:
        """
            Move a block from CPU to GPU. 
            Returns True if successful.
        """
        if block_id not in self.block_mapping:
            return False

        location, cpu_idx = self.block_mapping[block_id]
        if location == "gpu":
            return True

        # Find space in GPU
        gpu_free_idx = torch.where(~self.gpu_block_usage)[0]
        # No free space, try to evict a block
        if len(gpu_free_idx) == 0 and not self._evict_gpu_to_cpu():
            # If eviction to CPU failed, try direct LRU eviction from GPU
            gpu_idx = self._evict_lru(self.gpu_lru)
            if gpu_idx is None:
                # If all GPU blocks are pinned, we can't move to GPU
                return False

            # Now we have a free spot
            gpu_free_idx = torch.where(~self.gpu_block_usage)[0]
            if len(gpu_free_idx) == 0:
                # This should not happen, but just in case
                return False

        # Move the block using pinned memory copy
        gpu_idx = gpu_free_idx[0].item()

        # Use stream for async copy
        stream = self._get_next_stream()
        with torch.cuda.stream(stream):
            self.gpu_blocks[gpu_idx].copy_(
                self.cpu_blocks[cpu_idx], non_blocking=True
            )
        # Synchronize to ensure copy is complete
        stream.synchronize()

        self.gpu_block_usage[gpu_idx] = True

        # Update mappings
        self.block_mapping[block_id] = ("gpu", gpu_idx)
        if block_id not in self.pinned_blocks:
            self.gpu_lru.append(block_id)

        # Keep in CPU too as backup unless CPU is full
        if self.cpu_block_usage.sum() >= self.num_cpu_blocks * 0.9:  # 90% full
            self.cpu_block_usage[cpu_idx] = False
            if block_id in self.cpu_lru:
                self.cpu_lru.remove(block_id)

        return True

    def move_to_cpu(self, block_id: int) -> bool:
        """Move a block from GPU to CPU. Returns True if successful."""
        if block_id not in self.block_mapping:
            return False

        location, gpu_idx = self.block_mapping[block_id]
        if location == "cpu":
            return True

        # Find space in CPU
        cpu_free_idx = torch.where(~self.cpu_block_usage)[0]
        if len(cpu_free_idx) == 0:
            # No free space, evict LRU
            cpu_idx = self._evict_lru(self.cpu_lru)
            if cpu_idx is None:
                return False

            # Now we have a free spot
            cpu_free_idx = torch.where(~self.cpu_block_usage)[0]
            if len(cpu_free_idx) == 0:
                # This should not happen, but just in case
                return False

        # Move the block using pinned memory copy
        cpu_idx = cpu_free_idx[0].item()

        # Use stream for async copy
        stream = self._get_next_stream()
        with torch.cuda.stream(stream):
            self.cpu_blocks[cpu_idx].copy_(
                self.gpu_blocks[gpu_idx], non_blocking=True
            )
        # Synchronize to ensure copy is complete
        stream.synchronize()

        self.cpu_block_usage[cpu_idx] = True

        # Update mappings
        self.block_mapping[block_id] = ("cpu", cpu_idx)
        if block_id not in self.pinned_blocks:
            self.cpu_lru.append(block_id)

        # Free GPU space
        self.gpu_block_usage[gpu_idx] = False
        if block_id in self.gpu_lru:
            self.gpu_lru.remove(block_id)

        return True

    def multi_transfer(
        self, block_ids: list[int], source: str, target: str
    ) -> bool:
        """
        Transfer multiple blocks between CPU and GPU using multiple CUDA streams
        Returns True if all transfers were successful.

        Args:
            block_ids: List of block IDs to transfer
            source: Source location ('cpu' or 'gpu')
            target: Target location ('cpu' or 'gpu')
        """
        if source not in ["cpu", "gpu"] or target not in ["cpu", "gpu"]:
            raise ValueError("Source and target must be either 'cpu' or 'gpu'")

        if source == target:
            return True  # Nothing to do

        # Validate all blocks exist and are in the source location
        for block_id in block_ids:
            if block_id not in self.block_mapping:
                return False

            location, _ = self.block_mapping[block_id]
            if location != source:
                return False

        # Prepare transfer operations
        transfers = []
        for block_id in block_ids:
            if source == "cpu" and target == "gpu":
                success = self._prepare_cpu_to_gpu_transfer(block_id)
            else:  # gpu to cpu
                success = self._prepare_gpu_to_cpu_transfer(block_id)

            if success:
                transfers.append((block_id, success))

        # If no successful transfers prepared, return False
        if not transfers:
            return False

        # Execute transfers using multiple streams
        for i, (block_id, transfer_info) in enumerate(transfers):
            stream_idx = i % self.num_streams
            stream = self.streams[stream_idx]

            source_idx, target_idx = transfer_info
            if source == "cpu" and target == "gpu":
                with torch.cuda.stream(stream):
                    self.gpu_blocks[target_idx].copy_(
                        self.cpu_blocks[source_idx], non_blocking=True
                    )
            else:  # gpu to cpu
                with torch.cuda.stream(stream):
                    self.cpu_blocks[target_idx].copy_(
                        self.gpu_blocks[source_idx], non_blocking=True
                    )

        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()

        # Update mappings and LRU lists
        for block_id, (source_idx, target_idx) in transfers:
            if source == "cpu" and target == "gpu":
                # Update mappings
                self.block_mapping[block_id] = ("gpu", target_idx)
                if block_id not in self.pinned_blocks:
                    if block_id in self.cpu_lru:
                        self.cpu_lru.remove(block_id)
                    self.gpu_lru.append(block_id)

                # Keep in CPU too as backup unless CPU is full
                if (
                    self.cpu_block_usage.sum() >= self.num_cpu_blocks * 0.9
                ):  # 90% full
                    self.cpu_block_usage[source_idx] = False

            else:  # gpu to cpu
                # Update mappings
                self.block_mapping[block_id] = ("cpu", target_idx)
                if block_id not in self.pinned_blocks:
                    if block_id in self.gpu_lru:
                        self.gpu_lru.remove(block_id)
                    self.cpu_lru.append(block_id)

                # Free GPU space
                self.gpu_block_usage[source_idx] = False

        self.stats["multi_transfers"] += 1
        return True

    def _prepare_cpu_to_gpu_transfer(self, block_id):
        """Prepare a CPU to GPU transfer by allocating GPU space."""
        _, cpu_idx = self.block_mapping[block_id]

        # Find space in GPU
        gpu_free_idx = torch.where(~self.gpu_block_usage)[0]
        # No free space, try to evict a block
        if len(gpu_free_idx) == 0 and not self._evict_gpu_to_cpu():
            # If eviction to CPU failed, try direct LRU eviction from GPU
            gpu_idx = self._evict_lru(self.gpu_lru)
            if gpu_idx is None:
                # If all GPU blocks are pinned, we can't move to GPU
                return False

            # Now we have a free spot
            gpu_free_idx = torch.where(~self.gpu_block_usage)[0]
            if len(gpu_free_idx) == 0:
                # This should not happen, but just in case
                return False

        # Mark GPU block as used
        gpu_idx = gpu_free_idx[0].item()
        self.gpu_block_usage[gpu_idx] = True

        return (cpu_idx, gpu_idx)

    def _prepare_gpu_to_cpu_transfer(self, block_id):
        """Prepare a GPU to CPU transfer by allocating CPU space."""
        _, gpu_idx = self.block_mapping[block_id]

        # Find space in CPU
        cpu_free_idx = torch.where(~self.cpu_block_usage)[0]
        if len(cpu_free_idx) == 0:
            # No free space, evict LRU
            cpu_idx = self._evict_lru(self.cpu_lru)
            if cpu_idx is None:
                return False

            # Now we have a free spot
            cpu_free_idx = torch.where(~self.cpu_block_usage)[0]
            if len(cpu_free_idx) == 0:
                # This should not happen, but just in case
                return False

        # Mark CPU block as used
        cpu_idx = cpu_free_idx[0].item()
        self.cpu_block_usage[cpu_idx] = True

        return (gpu_idx, cpu_idx)

    def get_stats(self):
        """Return cache statistics."""
        stats = self.stats.copy()
        stats.update(
            {
                "gpu_usage": self.gpu_block_usage.sum().item()
                / self.num_gpu_blocks,
                "cpu_usage": self.cpu_block_usage.sum().item()
                / self.num_cpu_blocks,
                "gpu_blocks_used": self.gpu_block_usage.sum().item(),
                "cpu_blocks_used": self.cpu_block_usage.sum().item(),
                "total_blocks": len(self.block_mapping),
                "num_streams": self.num_streams,
            }
        )
        return stats
