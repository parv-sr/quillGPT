from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from config import Config
from model.gpt import GPT


class ONNXTester:
    def __init__(self, checkpoint_path: str, onnx_path: str, config: Config) -> None:
        self.checkpoint_path = checkpoint_path
        self.onnx_path = onnx_path
        self.config = config

    def load_pytorch_model(self) -> GPT:
        model = GPT(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_context=self.config.max_context,
            dropout=self.config.dropout
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(checkpoint)
        model.eval()

        return model
    
    def load_onnx_model(self) -> ort.InferenceSession:
        return ort.InferenceSession(
            str(self.onnx_path),
            providers=[
                "CPUExecutionProvider"
            ]
        )
    
    def generate_test_input(self) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(1, self.config.max_context),
            dtype=torch.long
        )
    
    def test(self) -> None:
        print("loading pytorch model...")
        pytorch_model = self.load_pytorch_model()
        print("loading ONNX model...")
        session = self.load_onnx_model()

        print("\nONNX Inputs: ")

        for input_info in session.get_inputs():
            print(f"Name: {input_info.name}")
            print(f"Shape: {input_info.shape}")
            print(f"Type: {input_info.type}")

        print(f"\nONNX Outputs: ")

        for output_info in session.get_outputs():
            print(f"Name: {output_info.name}")
            print(f"Shape: {output_info.shape}")
            print(f"Type: {output_info.shape}")

        tokens = self.generate_test_input()

        with torch.no_grad():
            pytorch_logits = pytorch_model(
                tokens
            )

        onnx_inputs = {
            "tokens" : tokens.numpy()
        }

        onnx_outputs = session.run(
            ["logits"],
            onnx_inputs
        )

        onnx_logits = torch.from_numpy(onnx_outputs[0])

        print(
            f"\nPyTorch output shape: "
            f"{pytorch_logits.shape}"
        )

        print(
            f"ONNX output shape: "
            f"{onnx_logits.shape}"
        )

        difference = torch.abs(
            pytorch_logits - onnx_logits
        )

        max_difference = difference.max().item()
        mean_difference = difference.mean().item()

        print(
            f"\nMaximum absolute difference: "
            f"{max_difference:.8f}"
        )

        print(
            f"Mean absolute difference: "
            f"{mean_difference:.8f}"
        )

        is_close = torch.allclose(
            pytorch_logits,
            onnx_logits,
            atol=1e-4,
            rtol=1e-3,
        )

        print(
            f"\nNumerically equivalent: "
            f"{is_close}"
        )

        if not is_close:
            raise RuntimeError(
                "PyTorch and ONNX outputs differ "
                "beyond the allowed tolerance."
            )

        print(
            "\nONNX verification successful."
        )


def main() -> None:

    config = Config()

    tester = ONNXTester(
        checkpoint_path=(
            "artifacts/models/"
            "tinygpt-v0.0.1.pth"
        ),
        onnx_path=(
            "artifacts/models/"
            "tinygpt-v0.0.1.onnx"
        ),
        config=config,
    )

    tester.test()


if __name__ == "__main__":
    main()

