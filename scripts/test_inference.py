from config import Config

from data.corpus import TextCorpus
from data.tokenizer import CharacterTokenizer

from backend.inference.engine import (
    ONNXInferenceEngine,
)

from backend.inference.generator import (
    TextGenerator,
)


def main() -> None:

    config = Config()

    # Load corpus and tokenizer
    corpus = TextCorpus(
        "data/raw/input.txt"
    )

    tokenizer = CharacterTokenizer(
        corpus.text
    )

    # Load ONNX model
    engine = ONNXInferenceEngine(
        "artifacts/models/tinygpt-v0.0.1.onnx"
    )

    print(
        f"Execution providers: "
        f"{engine.providers}"
    )

    # Build generator
    generator = TextGenerator(
        engine=engine,
        tokenizer=tokenizer,
        max_context=config.max_context,
    )

    # Our current ONNX model expects exactly
    # 128 tokens, so use a 128-character prompt.
    prompt = (
        "First Citizen:\n"
        "Before we proceed any further, "
        "hear me speak.\n\n"
        "All:\n"
        "Speak, speak.\n\n"
        "First Citizen:\n"
        "You are all resolved rather to die "
        "than to famish?\n\n"
    )

    print("\nPrompt:")
    print(prompt)

    generated = generator.generate(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.8,
    )

    print("\nGenerated:")
    print(generated)


if __name__ == "__main__":
    main()