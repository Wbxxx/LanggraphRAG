import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 现在可以用 os.getenv() 或 os.environ 获取环境变量
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

