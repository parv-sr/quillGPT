from pathlib import Path

import numpy as np
import onnxruntime as ort

from typing import List


class ONNXInferenceEngine:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "ONNX model not found"
            )
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self._get_providers()
        )

        self.input_name = (self.session.get_inputs()[0].name)
        self.output_name = (self.session.get_outputs()[0].name)

    @staticmethod
    def _get_providers() -> List[str]:
        available = (ort.get_available_providers())

        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        return ["CPUExecutionProvider"]
    
    @property
    def providers(self) -> List[str]:
        return self.session.get_providers()
    
    def predict(self, tokens: np.ndarray) -> np.ndarray:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        
        if tokens.dtype != np.int64:
            tokens = tokens.astype(np.int64)

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: tokens
            }
        )

        return outputs[0]