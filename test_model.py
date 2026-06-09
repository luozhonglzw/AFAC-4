"""测试百炼 API 可用的模型名称"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

models_to_test = ["qwen-plus", "qwen3.6-plus", "qwen-plus-2025-07-28"]

for model in models_to_test:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "你好，请回复'测试成功'"}],
            max_tokens=10
        )
        print(f"[OK] {model}: {response.choices[0].message.content}")
        print(f"     prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
    except Exception as e:
        print(f"[FAIL] {model}: {str(e)[:120]}")
