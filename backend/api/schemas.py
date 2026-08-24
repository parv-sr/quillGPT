from pydantic import BaseModel, Field

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=100, ge=1, le=1000)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)

class InferenceResponse(BaseModel):
    text: str