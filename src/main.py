
from fastapi import FastAPI
from core.config import OLLAMA_DEFAULT_MODEL, VLLM_DEFAULT_MODEL
from api.v1.routers import llm_router

app = FastAPI(
    title="Hybrid LLM Gateway",
    description="A FastAPI gateway for Ollama and vLLM inference.",
    version="0.1.0"
)

app.include_router(llm_router.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome! Use /api/v1/llm/generate to start."}
