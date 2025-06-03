
from pydantic import BaseModel, Field
from typing import Optional

class LLMRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for the LLM.")
    max_tokens: int = Field(150, description="Maximum number of tokens to generate.")
    temperature: float = Field(0.7, description="Sampling temperature for generation.")
    model: Optional[str] = Field(None, description="Specific model to use (overrides default).")
    engine: Optional[str] = Field("ollama", description="Which LLM engine to use: 'ollama' or 'vllm'.")

class LLMResponse(BaseModel):
    generated_text: str
    model_used: str
    engine_used: str