# SPDX-License-Identifier: Apache-2.0

import logging
from vllm import LLM, SamplingParams
from rich.console import Console
from rich.table import Table
from datetime import datetime
import os

# os.environ["VLLM_USE_SPARSE_OFFLOAD"] = 1
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.makedirs("logs", exist_ok=True)
log_filename = datetime.now().strftime("logs/vllm_%Y%m%d_%H%M%S.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("vllm_server")

console = Console()

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
model = "meta-llama/Llama-3.2-1B-Instruct"
console.print("[bold magenta] Loading model... [/bold magenta]")
logger.info(f"Loading model: {model}")
llm = LLM(model=model, enforce_eager=True, gpu_memory_utilization=0.7)
logger.info("Loading model successful.")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

table = Table(title="📜 vLLM generated texts", show_lines=True)
table.add_column("Prompt", justify="left", style="cyan", no_wrap=False)
table.add_column("Generated Text", justify="left", style="green")

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text.strip()
    table.add_row(prompt, generated_text)
    logger.info(f"Prompt: {prompt} -> Generated: {generated_text}")

console.print(table)
logger.info("Finished generating texts.")