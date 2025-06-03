# Hybrid LLM Gateway

FastAPI gateway for Ollama/vLLM inference. Unifies API, allows engine/model selection, and handles errors efficiently.

## Quickstart (Linux)

1.  **Clone**: `git clone https://github.com/fathurwithyou/simple-chatbot-engine.git && cd simple-chatbot-engine/src`
2.  **Install `uv`**: `curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.cargo/bin:$PATH"`
3.  **Setup Env**: `uv venv && source .venv/bin/activate`
4.  **Install Dependencies**: `uv sync`
5.  **Configure `.env`**: Create `.env` in `src` (e.g., `OLLAMA_BASE_URL="http://localhost:11434/api"`, `VLLM_API_URL="http://localhost:8000/generate"`, etc.).
6.  **Run Services**: Start Ollama/vLLM.
    * **Ollama**: `ollama serve` (or it runs automatically) and `ollama pull tinyllama`
    * **vLLM**: `uv run python -m vllm.entrypoints.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host 0.0.0.0 --port 8000`
7.  **Run Gateway**: `uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload`

## API Endpoint

* `POST /api/v1/llm/generate`
    * **Request**: `{"prompt": "...", "engine": "ollama" | "vllm", "max_tokens": 150, "temperature": 0.7, "model": "..."}`
    * **Response**: `{"generated_text": "...", "model_used": "...", "engine_used": "..."}`

### Example `curl` Commands

To test the API, ensure your FastAPI gateway is running (`http://localhost:8001`), and the respective LLM service (Ollama or vLLM) is active with the specified model.

#### Using Ollama Engine

```bash
curl -X POST "http://localhost:8001/api/v1/llm/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Jelaskan konsep fisika kuantum dalam tiga kalimat.",
           "max_tokens": 200,
           "temperature": 0.7,
           "engine": "ollama",
           "model": "tinyllama"
         }'
```
#### Using vLLM Engine

```bash
curl -X POST "http://localhost:8001/api/v1/llm/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Buatlah daftar 5 manfaat belajar bahasa pemrograman baru.",
           "max_tokens": 150,
           "temperature": 0.8,
           "engine": "vllm",
           "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
         }'
```
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`: `uv run python -m vllm.entrypoints.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host 0.0.0.0 --port 8000`)*

## Error Handling

Handles `400 Invalid Engine`, `404 Model Not Found`, and `503 Service Unavailable`.
