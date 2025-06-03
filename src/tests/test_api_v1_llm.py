import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from main import app
from unittest.mock import patch, AsyncMock
import json
from core.exceptions import LLMServiceError, ModelNotFoundError, InvalidEngineError

@pytest_asyncio.fixture
async def async_client():
    """
    Fixture to create an AsyncClient for testing the FastAPI application.
    Uses ASGITransport to test the app directly without a running server.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Test the root endpoint."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome! Use /api/v1/llm/generate to start."}


@pytest.mark.asyncio
@patch('api.v1.routers.llm_router.generate_with_ollama') 
async def test_generate_ollama_success(mock_generate_with_ollama: AsyncMock, async_client: AsyncClient):
    """Test successful generation with Ollama engine"""
    mock_generate_with_ollama.return_value = "Any response from Ollama"

    response = await async_client.post(
        "/api/v1/llm/generate",
        json={
            "prompt": "Hello Ollama",
            "engine": "ollama",
            "model": "tinyllama"
        }
    )
    
    assert response.status_code == 200
    response_data = response.json()
    
    assert isinstance(response_data["generated_text"], str)
    assert len(response_data["generated_text"]) > 0  
    assert response_data["model_used"] == "tinyllama"
    assert response_data["engine_used"] == "ollama"
    
    mock_generate_with_ollama.assert_called_once()

@pytest.mark.asyncio
@patch('api.v1.routers.llm_router.generate_with_vllm') 
async def test_generate_vllm_success(mock_generate_with_vllm: AsyncMock, async_client: AsyncClient):
    """Test successful generation with vLLM engine."""
    mock_generate_with_vllm.return_value = "This is a generated response from vLLM."

    response = await async_client.post(
        "/api/v1/llm/generate",
        json={
            "prompt": "Hello vLLM",
            "engine": "vllm",
            "model": "mistralai/Mistral-7B-Instruct-v0.2"
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        "generated_text": "This is a generated response from vLLM.",
        "model_used": "mistralai/Mistral-7B-Instruct-v0.2",
        "engine_used": "vllm"
    }
    mock_generate_with_vllm.assert_called_once()


@pytest.mark.asyncio
async def test_generate_invalid_engine(async_client: AsyncClient):
    """Test request with an invalid engine type."""
    response = await async_client.post(
        "/api/v1/llm/generate",
        json={
            "prompt": "Invalid engine test",
            "engine": "unknown_engine"
        }
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid engine specified: 'unknown_engine'. Choose 'ollama' or 'vllm'."
    }


@pytest.mark.asyncio
@patch('api.v1.routers.llm_router.generate_with_ollama') 
async def test_generate_ollama_service_error(mock_generate_with_ollama: AsyncMock, async_client: AsyncClient):
    """Test error handling when Ollama service fails."""
    mock_generate_with_ollama.side_effect = LLMServiceError(
        service_name="Ollama", original_detail="Connection refused"
    )

    response = await async_client.post(
        "/api/v1/llm/generate",
        json={
            "prompt": "Test Ollama error",
            "engine": "ollama",
            "max_tokens": 200,
            "temperature": 0.75,
            "model": "tinyllama"
        }
    )
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Error interacting with Ollama service: Connection refused"
    }
    mock_generate_with_ollama.assert_called_once()


@pytest.mark.asyncio
@patch('api.v1.routers.llm_router.generate_with_vllm')
async def test_generate_vllm_model_not_found(mock_generate_with_vllm: AsyncMock, async_client: AsyncClient):
    """Test error handling when vLLM model is not found."""
    mock_generate_with_vllm.side_effect = ModelNotFoundError(
        model_name="non_existent_vllm_model", engine_name="vLLM"
    )

    response = await async_client.post(
        "/api/v1/llm/generate",
        json={
            "prompt": "Test vLLM model not found",
            "engine": "vllm",
            "model": "non_existent_vllm_model"
        }
    )
    assert response.status_code == 404
    assert response.json() == {
        "detail": "Model 'non_existent_vllm_model' not found on vLLM."
    }
    mock_generate_with_vllm.assert_called_once()


@pytest.mark.asyncio
async def test_generate_missing_prompt(async_client: AsyncClient):
    """Test request with missing required 'prompt' field."""
    response = await async_client.post(
        "/api/v1/llm/generate",
        json={
            "engine": "ollama"
        }
    )
    assert response.status_code == 422
    assert "prompt" in response.json()["detail"][0]["loc"]
    assert response.json()["detail"][0]["msg"] == "Field required"