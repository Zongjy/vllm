from vllm import LLM, SamplingParams
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import box
import logging
import os
from datetime import datetime
import sys

# 创建日志目录
os.makedirs("logs", exist_ok=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置基础日志
log_filename = datetime.now().strftime("logs/vllm_%Y%m%d_%H%M%S.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# 获取 vLLM 专用日志器并设置级别
vllm_logger = logging.getLogger("vllm")
vllm_logger.setLevel(logging.DEBUG)  # 更详细的日志级别

# 初始化 rich console
console = Console()

model_name = "Qwen/Qwen2.5-0.5B-Instruct"


# 加载本地模型
console.print("[bold magenta]🚀 正在加载模型...[/bold magenta]")
vllm_logger.info(
    f"开始加载模型：./{model_name}"  # noqa: G004
)  # 记录模型加载日志  # noqa: G004

try:
    llm = LLM(
        model=f"{model_name}", max_model_len=16384, gpu_memory_utilization=0.5
    )
    vllm_logger.info("模型加载成功")
except Exception as e:
    log_str = f"模型加载失败: {str(e)}"
    vllm_logger.critical(log_str)
    console.print("[bold red]❌ 模型加载失败，请检查模型路径[/bold red]")
    sys.exit(1)

console.print(
    f"[bold green]✅ 模型加载完成![/bold green] [dim](日志文件: \
          {log_filename})[/dim]\n"
)

# 设置生成参数
sampling_params = SamplingParams(temperature=0.8, max_tokens=200)
log_str = f"初始化采样参数: {sampling_params}"
vllm_logger.debug(log_str)

# 创建对话历史表格
history_table = Table(
    title="[bold cyan]🤖 对话历史[/bold cyan]",
    box=box.ROUNDED,
    header_style="bold blue",
    show_lines=True,
    expand=True,
)
history_table.add_column("轮次", width=8)
history_table.add_column("问题", width=40)
history_table.add_column("回答", width=80)

# ...后续代码保持不变...

round_count = 1

# 交互循环
while True:
    try:
        # 美化输入提示
        prompt = Prompt.ask(
            "[bold yellow]💬 请输入你的问题[/bold yellow] (输入 'exit' 退出)",
            console=console,
        )

        if prompt.__len__() == 0:
            console.print("[red]❌ 请输入有效的问题")
            continue

        if prompt.lower() in ("exit", "quit"):
            console.print("[bold magenta]\n👋 感谢使用，再见！[/bold magenta]")
            break

        # 生成回答
        with console.status("[bold green]生成中..."):
            outputs = llm.generate([prompt], sampling_params)

        # 提取结果
        answer = outputs[0].outputs[0].text.strip()
        for j in range(0, len(outputs)):
            for i in range(1, len(outputs[0].outputs)):
                answer += f"\n{outputs[j].outputs[i].text.strip()}"

        # 更新对话历史表格
        history_table.add_row(
            str(round_count),
            f"[italic]{prompt}[/italic]",
            f"[green]{answer}[/green]",
        )

        # 打印本次回答
        console.print("\n[bold cyan]📝 本次回答：[/bold cyan]")
        console.print(f"[green]{answer}[/green]\n")

        # 打印完整对话历史
        console.print(history_table)
        round_count += 1

    except EOFError:
        console.print("[bold magenta]\n👋 读取到EOF，程序退出[/bold magenta]")
        break
    except KeyboardInterrupt:
        console.print("[bold magenta]\n👋 用户中断，程序退出[/bold magenta]")
        break
