#!/bin/bash

export VLLM_USE_V1=1
# export VLLM_USE_SPARSE_OFFLOAD=1

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --max-model-len 7200 \
    --gpu-memory-utilization 0.99 \
    --enforce-eager \
    --trust_remote_code