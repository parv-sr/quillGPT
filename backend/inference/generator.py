from typing import Tuple

import numpy as np

from data.bpe_tokenizer import BPETokenizer

from .engine import (
    KeyValueCaches,
    ONNXInferenceEngine
)


class TextGenerator:
    def __init__(
        self,
        engine: ONNXInferenceEngine,
        tokenizer: BPETokenizer,
        max_context: int
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_context = max_context

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float
    ) -> int:
        if temperature < 0:
            raise ValueError(
                "Temperature cannot be negative."
            )

        if temperature == 0:
            return int(np.argmax(logits))

        logits = logits / temperature
        logits = logits - np.max(logits)

        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        return int(
            np.random.choice(
                len(probabilities),
                p=probabilities
            )
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8
    ) -> str:
        token_ids = self.tokenizer.encode(prompt)

        if not token_ids:
            raise ValueError(
                "Prompt cannot be empty."
            )

        if len(token_ids) > self.max_context:
            token_ids = token_ids[-self.max_context:]

        tokens = np.asarray(
            [token_ids],
            dtype=np.int64
        )

        logits, past_key_values = self.engine.prefill(
            tokens
        )

        next_token = self._sample(
            logits[0, -1, :],
            temperature
        )

        token_ids.append(next_token)

        # Decode one token at a time using the KV cache.
        for _ in range(max_new_tokens - 1):
            token = np.asarray(
                [[next_token]],
                dtype=np.int64
            )

            logits, past_key_values = self.engine.decode(
                token,
                past_key_values
            )

            next_token = self._sample(
                logits[0, -1, :],
                temperature
            )

            token_ids.append(next_token)

        return self.tokenizer.decode(
            token_ids=token_ids
        )