
from fastapi import HTTPException, status
from typing import Optional


class APIException(HTTPException):
    """Base custom exception for API errors."""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "An unexpected error occurred.",
        headers: Optional[dict[str, str]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class InvalidEngineError(APIException):
    """Exception raised when an invalid LLM engine is specified."""

    def __init__(self, engine_name: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid engine specified: '{engine_name}'. Choose 'ollama' or 'vllm'."
        )


class LLMServiceError(APIException):
    """Exception raised when there's an error interacting with an LLM service (Ollama/vLLM)."""

    def __init__(self, service_name: str, original_detail: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error interacting with {service_name} service: {original_detail}"
        )


class ModelNotFoundError(APIException):
    """Exception raised when the requested model is not found on the specified engine."""

    def __init__(self, model_name: str, engine_name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found on {engine_name}."
        )
