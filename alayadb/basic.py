# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
import vllm.logger
import os

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_USE_SPARSE_OFFLOAD"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.makedirs("logs", exist_ok=True)

logger = vllm.logger.logger

# Sample prompts.
prompts = [
    "Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is Hello, my name is ",
    "The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is ",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
model = "meta-llama/Llama-3.2-1B-Instruct"
llm = LLM(model=model, enforce_eager=True, gpu_memory_utilization=0.7)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text.strip()
    logger.info_once(f"Prompt: {prompt} -> Generated: {generated_text}")

logger.info_once("Finished generating texts.")
