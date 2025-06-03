
from fastapi import APIRouter, status
from api.v1.models import LLMRequest, LLMResponse
from services.ollama_service import generate_with_ollama
from services.vllm_service import generate_with_vllm
from core.config import OLLAMA_DEFAULT_MODEL, VLLM_DEFAULT_MODEL
from core.exceptions import InvalidEngineError, APIException

router = APIRouter(
    prefix="/llm",
    tags=["LLM Generation"],
)

@router.post("/generate", response_model=LLMResponse)
async def generate_text(request: LLMRequest):
    """
    Generates text using either Ollama or vLLM based on the 'engine' specified.
    """
    if request.engine == "ollama":
        model_to_use = request.model if request.model else OLLAMA_DEFAULT_MODEL
        generated_text = await generate_with_ollama(request)
        return LLMResponse(
            generated_text=generated_text,
            model_used=model_to_use,
            engine_used="ollama"
        )
    elif request.engine == "vllm":
        model_to_use = request.model if request.model else VLLM_DEFAULT_MODEL
        generated_text = await generate_with_vllm(request)
        return LLMResponse(
            generated_text=generated_text,
            model_used=model_to_use,
            engine_used="vllm"
        )
    else:
        raise InvalidEngineError(engine_name=request.engine)
