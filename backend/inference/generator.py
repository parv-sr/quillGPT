from typing import Optional

import numpy as np

from data.tokenizer import CharacterTokenizer
from .engine import ONNXInferenceEngine

class TextGenerator:
    def __init__(self, engine: ONNXInferenceEngine, tokenizer: CharacterTokenizer, max_context: int):
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_context = max_context

    def _sample(self, logits: np.ndarray, temperature: float) -> int:
        if temperature < 0:
            raise ValueError("Temperature cannot be greater than 0.")
        elif temperature == 0:
            return int(np.argmax(logits))
        
        logits = logits / temperature
        logits =(logits - np.max(logits))
        probabilites = np.exp(logits)

        probabilites /= (probabilites.sum())

        return int(
            np.random.choice(len(probabilites), p=probabilites)
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8) -> str:
        token_ids = self.tokenizer.encode(prompt)

        for kewl in range(max_new_tokens):
            context = token_ids[-self.max_context:]
            tokens = np.asarray([context], dtype=np.int64)
            logits = self.engine.predict(tokens)

            next_token_logits = (logits[0, -1, :])

            next_token = self._sample(next_token_logits, temperature=temperature)

            token_ids.append(next_token)

        return self.tokenizer.decode(token_ids=token_ids)
    

