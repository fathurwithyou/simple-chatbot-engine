import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/generate")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama2")
VLLM_DEFAULT_MODEL = os.getenv("VLLM_DEFAULT_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")