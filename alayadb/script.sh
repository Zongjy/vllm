#!/bin/bash

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [{
            "role": "user",
            "content": "Tell me about the history of the Lofi genre."
        }],
    }'