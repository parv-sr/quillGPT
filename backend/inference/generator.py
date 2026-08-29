from typing import List

import numpy as np

from data.bpe_tokenizer import BPETokenizer

from .engine import (
    KeyValueCaches,
    ONNXInferenceEngine
)


class TextGenerator:
    def __init__(self, engine: ONNXInferenceEngine, tokenizer: BPETokenizer, max_context: int) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_context = max_context

    def _apply_repetition_penalty(self, logits: np.ndarray, token_ids: List[int], repetition_penalty: float) -> np.ndarray:
        if repetition_penalty < 1.0:
            raise ValueError("Repetition penalty must be at least 1.0.")

        if repetition_penalty == 1.0:
            return logits

        logits = logits.copy()

        for token_id in set(token_ids):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

        return logits

    def _sample(self, logits: np.ndarray, token_ids: List[int], temperature: float, top_p: float, repetition_penalty: float) -> int:
        if temperature < 0:
            raise ValueError("Temperature cannot be negative.")

        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1.0.")

        logits = self._apply_repetition_penalty(logits, token_ids, repetition_penalty)

        if temperature == 0:
            return int(np.argmax(logits))

        logits = logits / temperature
        logits = logits - np.max(logits)

        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        if top_p < 1.0:
            sorted_indices = np.argsort(probabilities)[::-1]
            sorted_probabilities = probabilities[sorted_indices]

            cumulative_probabilities = np.cumsum(sorted_probabilities)

            cutoff = cumulative_probabilities > top_p

            if np.any(cutoff):
                first_cutoff = np.argmax(cutoff)
                cutoff[first_cutoff] = False

            probabilities[sorted_indices[cutoff]] = 0.0
            probabilities /= probabilities.sum()

        return int(np.random.choice(len(probabilities), p=probabilities))

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.9, repetition_penalty: float = 1.15) -> str:
        token_ids = self.tokenizer.encode(prompt)

        if not token_ids:
            raise ValueError("Prompt cannot be empty.")

        if len(token_ids) > self.max_context:
            token_ids = token_ids[-self.max_context:]

        tokens = np.asarray([token_ids], dtype=np.int64)

        logits, past_key_values = self.engine.prefill(tokens)

        next_token = self._sample(
            logits[0, -1, :],
            token_ids,
            temperature,
            top_p,
            repetition_penalty
        )

        token_ids.append(next_token)

        for _ in range(max_new_tokens - 1):
            token = np.asarray([[next_token]], dtype=np.int64)

            logits, past_key_values = self.engine.decode(
                token,
                past_key_values
            )

            next_token = self._sample(
                logits[0, -1, :],
                token_ids,
                temperature,
                top_p,
                repetition_penalty
            )

            token_ids.append(next_token)

        return self.tokenizer.decode(token_ids=token_ids)