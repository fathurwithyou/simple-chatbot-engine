
import httpx
from core.config import OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL
from api.v1.models import LLMRequest
from core.exceptions import LLMServiceError, ModelNotFoundError
from fastapi import status


async def generate_with_ollama(request: LLMRequest) -> str:
    model_name = request.model if request.model else OLLAMA_DEFAULT_MODEL
    url = f"{OLLAMA_BASE_URL}/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": request.prompt,
        "options": {
            "num_predict": request.max_tokens,
            "temperature": request.temperature,
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=600.0)
            response.raise_for_status()
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = httpx.json.loads(line)

                    if "error" in data and "model not found" in data["error"].lower():
                        raise ModelNotFoundError(
                            model_name=model_name, engine_name="Ollama")
                    if "response" in data:
                        full_response += data["response"]
                    if data.get("done"):
                        break
            return full_response.strip()
        except httpx.RequestError as exc:

            raise LLMServiceError(service_name="Ollama",
                                  original_detail=str(exc))
        except httpx.HTTPStatusError as exc:

            if exc.response.status_code == status.HTTP_404_NOT_FOUND:
                raise ModelNotFoundError(
                    model_name=model_name, engine_name="Ollama")
            raise LLMServiceError(service_name="Ollama",
                                  original_detail=exc.response.text)
        except Exception as exc:

            raise LLMServiceError(service_name="Ollama",
                                  original_detail=f"Unexpected error: {exc}")
