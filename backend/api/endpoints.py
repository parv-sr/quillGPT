from fastapi import APIRouter, Request
from backend.api.schemas import InferenceRequest, InferenceResponse

from typing import Dict, Any

router = APIRouter(prefix="/infer")

@router.post("/generate", response_model=InferenceResponse)
async def infer(infer_req: InferenceRequest, request: Request) -> Dict[str, Any]:
    generator = request.app.state.generator

    result = generator.generate(
        prompt=infer_req.prompt,
        max_new_tokens=infer_req.max_new_tokens,
        temperature=infer_req.temperature,
        top_p=infer_req.top_p,
        repetition_penalty=infer_req.repetition_penalty
    )

    return InferenceResponse(
        text=result
    )