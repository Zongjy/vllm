import logging
from openai import OpenAI
from rich.console import Console
from datetime import datetime
import os

os.makedirs("logs", exist_ok=True)
log_filename = datetime.now().strftime("logs/openai_client_%Y%m%d_%H%M%S.log")

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("openai_client")

# Rich Console
console = Console()

# Set API Key & Base URL
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

console.print("[bold yellow] Sending Request to vLLM Server...[/bold yellow]")
logger.info("Sending Request to vLLM Server.")

model = "meta-llama/Llama-3.2-1B-Instruct"
try:
    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Tell me a joke."},
        ],
    )
    response_text = chat_response.choices[0].message.content

    console.print(f"[bold green] Server Response:[/bold green] {response_text}")
    logger.info(f"Server Response: {response_text}")

except Exception as e:
    logger.error(f"Request failed: {str(e)}")
    console.print(f"[bold red] Request failed: {str(e)}[/bold red]")
