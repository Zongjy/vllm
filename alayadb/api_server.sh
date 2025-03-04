#!/bin/bash

expert VLLM_USE_V1=1
expert VLLM_USE_SPARSE_OFFLOAD=1

vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \