import torch
from vllm.attention.backends.abstract import AttentionType
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.model_executor.models.utils import extract_layer_index
from collections import defaultdict
from collections.abc import Sequence
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generic, List,
                    Optional, TypeVar, Union, overload)

if TYPE_CHECKING:
    from vllm.attention.layer import Attention

class KVCacheContextManager:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.device = torch.device("cpu")
        self.kv_caches: List[torch.Tensor] = []
        self.kv_cache_dict: Dict[str, torch.Tensor] = {}

    def _initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        if len(kv_cache_config.groups) > 1:
            raise NotImplementedError("Hybrid models with more than one KV"
                                      "cache type are not supported yet.")

        for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
            tensor_config = kv_cache_config.tensors[layer_name]
            assert tensor_config.size % layer_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // layer_spec.page_size_bytes
            if isinstance(layer_spec, FullAttentionSpec):
                kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
                    num_blocks, layer_spec.block_size, layer_spec.num_kv_heads,
                    layer_spec.head_size)
                dtype = layer_spec.dtype
                self.kv_cache_dict[layer_name] = torch.zeros(kv_cache_shape,
                                                             dtype=dtype,
                                                             device=self.device)
            else:
                raise NotImplementedError

    def _bind_kv_cache(self) -> None:
        index2name = defaultdict(list)
        for layer_name in self.kv_cache_dict:
            index2name[extract_layer_index(layer_name)].append(layer_name)

        for layer_index in sorted(index2name.keys()):
            layer_names = index2name[layer_index]
            if len(layer_names) > 1:
                raise NotImplementedError
            layer_name = layer_names[0]
            self.kv_caches.append(self.kv_cache_dict[layer_name])

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        for layer_name, kv_cache in self.kv_cache_dict.items():
            forward_ctx[layer_name].kv_cache = [kv_cache]

    def update_kvcache(
            self, 
            layer_name: str, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            slot_mapping: torch.Tensor,
            k_scale: float,
            v_scale: float
            ) -> None:
        """
        Update the kv cache with the given slot mapping.
        """
        # k_cpu = k.detach().cpu()
        # v_cpu = v.detach().cpu()

        # assert layer_name in self.kv_cache_dict

        # kv_cache = self.kv_cache_dict[layer_name]
        
        # kv_cache[0, slot_mapping, :] = k_cpu
        # kv_cache[1, slot_mapping, :] = v_cpu

        # del k_cpu, v_cpu
        # torch.cuda.empty_cache()
        # TODO: modified
        key_cache, value_cache = self.kv_cache_dict[layer_name].unbind(0)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            k_scale,
            v_scale,
        )

    def load_kvcache(
            self, 
            layer_name: str,
            gpu_device: torch.device) -> torch.Tensor:
        # TODO: modified
        return self.kv_cache_dict[layer_name].to(gpu_device)