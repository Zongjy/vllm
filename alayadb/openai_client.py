from openai import OpenAI
import vllm.logger
import os

os.makedirs("logs", exist_ok=True)

logger = vllm.logger.logger

# Set API Key & Base URL
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

logger.info_once("Sending Request to vLLM Server.")

model = "meta-llama/Llama-3.2-1B-Instruct"
try:
    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Tell me a joke."},
        ],
    )
    response_text = chat_response.choices[0].message.content

    logger.info_once(f"Server Response: {response_text}")

except Exception as e:
    logger.error(f"Request failed: {str(e)}")
