import os
import sys
from pathlib import Path

# repo root を import path に入れる（app.py / openai_gpt.py を直接 import する）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# app.py は import 時に env から Registry / client を組み立てるため、
# 実行環境の .env や shell env が test に漏れないよう最小構成へ固定する。
for _key in (
    "GPT_MODEL",
    "OPENAI_MODEL",
    "GPT_REASONING_EFFORT",
    "GPT_MAX_TOKENS",
    "LLM_MAX_TOKENS",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
):
    os.environ.pop(_key, None)

os.environ["OPENAI_API_KEY"] = "test-key-not-a-real-secret"
