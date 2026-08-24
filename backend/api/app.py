from fastapi import FastAPI
from contextlib import asynccontextmanager

from typing import Dict, Any

from config import Config
config = Config()

from backend.api import endpoints
from backend.inference.engine import ONNXInferenceEngine
from backend.inference.generator import TextGenerator
from data.tokenizer import CharacterTokenizer
from data.corpus import TextCorpus

MODEL_PATH: str = "artifacts/models/tinygpt-v0.0.1.onnx"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting inference gateway...")

    corpus = TextCorpus("data/raw/input.txt")
    tokenizer = CharacterTokenizer(corpus.text)
    engine = ONNXInferenceEngine(MODEL_PATH)

    generator = TextGenerator(
        engine=engine,
        tokenizer=tokenizer,
        max_context=config.max_context
    )

    app.state.tokenizer = tokenizer
    app.state.engine = engine
    app.state.generator = generator

    print("Inference engine loaded")

    yield

    print("Shutting down inference gateway...")

app = FastAPI(
    title="tinyGPT Inference gateway",
    lifespan=lifespan
)

app.include_router(router=endpoints.router, tags=["Inference"])

@app.get("/ping_health")
async def health() -> Dict[str, Any]:
    return {
        "model" : f"tinygpt-v{config.version}",
        "status" : "healthy"
    }