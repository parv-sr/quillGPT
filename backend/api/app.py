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

MODEL_PATH: str = f"artifacts/models/tinyGPT_v{config.version}.onnx"

@asynccontextmanager
async def lifespan(app: FastAPI):
    corpus = TextCorpus("data/raw")
    tokenizer = CharacterTokenizer(corpus.text)
    head_dim = config.embed_dim // config.num_heads
    engine = ONNXInferenceEngine(
        model_path=MODEL_PATH,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=head_dim
    )

    generator = TextGenerator(
        engine=engine,
        tokenizer=tokenizer,
        max_context=config.max_context
    )

    app.state.tokenizer = tokenizer
    app.state.engine = engine
    app.state.generator = generator

    yield

app = FastAPI(
    title="tinyGPT Inference gateway",
    lifespan=lifespan
)

app.include_router(router=endpoints.router, tags=["Inference"])

@app.get("/ping_health")
async def health() -> Dict[str, Any]:
    return {
        "model": f"tinygpt-v{config.version}",
        "status": "healthy"
    }