curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt": "San Francisco",
        "max_tokens": 7,
        "temperature": 0
    }'