from collections import OrderedDict
from typing import Union, Dict, List, Tuple
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec


class MemoryPool:
    def __init__(self, kv_cache_config, device="cuda"):
        """
            NOTE(liyi): The format of KVCacheConfig is:
                {
                "num_blocks": int,
                "tensors": { "layer_name": { "size": int }, ... },
                "groups": [["layer_name", ...], ...],
                "kv_cache_spec": { "layer_name": FullAttentionSpec, ... }
                }
            FullAttentionSpec is composed of: (num_blocks, block_size, 
            num_kv_heads, head_size, dtype)
            !OPTIMIZE(liyi):
            We store the KV cache for each layer in the cpu_pool, while 
            maintaining a unified gpu_pool. Currently, we just use the configs 
            determined by gpu memory to allocate kv_cache on cpu_pool, and use
            available_gpu_memory as the capacity of gpu_pool. We need refactor 
            kv_cache_configs and expand the capacity of cpu_pool.
        """
        self.kv_cache_config = kv_cache_config
        self.device = device

        if len(kv_cache_config.groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        self.cpu_pool: Dict[str, torch.Tensor] = {}
        total_blocks = 0
        for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
            tensor_config = kv_cache_config.tensors[layer_name]
            assert tensor_config.size % layer_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // layer_spec.page_size_bytes
            total_blocks += num_blocks
            if isinstance(layer_spec, FullAttentionSpec):
                kv_cache_shape = (2, num_blocks, layer_spec.block_size, 
                                layer_spec.num_kv_heads, layer_spec.head_size)
                dtype = layer_spec.dtype
                self.cpu_pool[layer_name] = torch.zeros(kv_cache_shape,
                                                    dtype=dtype,
                                                    device="cpu")
            else:
                raise NotImplementedError 

        kv_cache_shape = (2, total_blocks, layer_spec.block_size,
                            layer_spec.num_kv_heads, layer_spec.head_size)
        self.gpu_pool = torch.zeros(kv_cache_shape, dtype=dtype, device=device)

    def get(self, key: Union[str, int]):
        if key in self.gpu_pool:
            self.gpu_pool.move_to_end(key)
            return self.gpu_pool[key]

        if key in self.cpu_pool:
            return self._handle_cpu_hit(key)

        raise KeyError(f"Key {key} not found in memory pools")

    def _handle_cpu_hit(self, key: Union[str, int]):
        cpu_tensor = self.cpu_pool.pop(key)
        if len(self.gpu_pool) >= self.gpu_capacity:
            self._evict_from_gpu()
        gpu_tensor = cpu_tensor.to(self.device)

        self.gpu_pool[key] = gpu_tensor
        self.gpu_pool.move_to_end(key)
        return gpu_tensor

    def _evict_from_gpu(self):
        evict_key, evict_tensor = self.gpu_pool.popitem(last=False)
        if len(self.cpu_pool) >= self.cpu_capacity:
            self.cpu_pool.popitem(last=False)

        cpu_tensor = evict_tensor.cpu()
        self.cpu_pool[evict_key] = cpu_tensor
        self.cpu_pool.move_to_end(evict_key)

    def allocate(self, key: Union[str, int]):
        if key not in self.cpu_pool and key not in self.gpu_pool:
            new_tensor = torch.empty(self.page_size, device="cpu")
            self.cpu_pool[key] = new_tensor
            self.cpu_pool.move_to_end(key)

            if len(self.cpu_pool) > self.cpu_capacity:
                self.cpu_pool.popitem(last=False)
            return new_tensor
        raise KeyError(f"Key {key} already exists")

    def preload_to_gpu(self, key: Union[str, int]):
        if key in self.gpu_pool:
            return
        if key in self.cpu_pool:
            self._handle_cpu_hit(key)
        else:
            raise KeyError(f"Key {key} not found")