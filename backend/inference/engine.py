from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort


KeyValueCache = Tuple[np.ndarray, np.ndarray]
KeyValueCaches = Tuple[KeyValueCache, ...]


class ONNXInferenceEngine:
    def __init__(
        self,
        model_path: str,
        num_layers: int,
        num_heads: int,
        head_dim: int
    ) -> None:
        self.model_path = Path(model_path)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self._get_providers()
        )

        self.input_names = [
            input_.name
            for input_ in self.session.get_inputs()
        ]

        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

        self.token_input_name = self.input_names[0]

    @staticmethod
    def _get_providers() -> List[str]:
        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            return [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]

        return ["CPUExecutionProvider"]

    @property
    def providers(self) -> List[str]:
        return self.session.get_providers()

    @staticmethod
    def _prepare_tokens(
        tokens: np.ndarray
    ) -> np.ndarray:
        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape (batch, sequence)"
            )

        if tokens.dtype != np.int64:
            tokens = tokens.astype(np.int64)

        return tokens

    def _extract_cache(
        self,
        outputs: List[np.ndarray]
    ) -> KeyValueCaches:
        cache: List[KeyValueCache] = []

        for index in range(1, len(outputs), 2):
            cache.append(
                (
                    outputs[index],
                    outputs[index + 1]
                )
            )

        return tuple(cache)

    def _create_initial_cache(
        self,
        batch_size: int
    ) -> KeyValueCaches:
        cache: List[KeyValueCache] = []

        for _ in range(self.num_layers):
            key = np.zeros(
                (
                    batch_size,
                    self.num_heads,
                    1,
                    self.head_dim
                ),
                dtype=np.float32
            )

            value = np.zeros_like(key)

            cache.append((key, value))

        return tuple(cache)

    def prefill(
        self,
        tokens: np.ndarray
    ) -> Tuple[np.ndarray, KeyValueCaches]:
        tokens = self._prepare_tokens(tokens)

        inputs: Dict[str, np.ndarray] = {
            self.token_input_name: tokens
        }

        initial_cache = self._create_initial_cache(
            batch_size=tokens.shape[0]
        )

        for layer, (key, value) in enumerate(initial_cache):
            inputs[f"past_key_{layer}"] = key
            inputs[f"past_value_{layer}"] = value

        outputs = self.session.run(
            self.output_names,
            inputs
        )

        return (
            outputs[0],
            self._extract_cache(outputs)
        )

    def decode(
        self,
        token: np.ndarray,
        past_key_values: KeyValueCaches
    ) -> Tuple[np.ndarray, KeyValueCaches]:
        token = self._prepare_tokens(token)

        if token.shape[1] != 1:
            raise ValueError(
                "decode expects exactly one token"
            )

        if len(past_key_values) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} KV caches, "
                f"received {len(past_key_values)}"
            )

        inputs: Dict[str, np.ndarray] = {
            self.token_input_name: token
        }

        for layer, (key, value) in enumerate(past_key_values):
            inputs[f"past_key_{layer}"] = key
            inputs[f"past_value_{layer}"] = value

        outputs = self.session.run(
            self.output_names,
            inputs
        )

        return (
            outputs[0],
            self._extract_cache(outputs)
        )