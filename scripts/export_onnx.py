from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from config import Config
from model.gpt import GPT


class ONNXGPTWrapper(nn.Module):
    def __init__(self, model: GPT, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers
    
    def forward(self, tokens: torch.Tensor, *past_key_values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        cache = tuple(
            (past_key_values[i], past_key_values[i+1]) for i in range(0, len(past_key_values), 2)
        )

        logits, present_key_values = self.model(
            tokens, past_key_values=cache, use_cache=True
        )

        outputs = [logits]

        for key, value in present_key_values:
            outputs.append(key)
            outputs.append(value)
        
        return tuple(outputs)


class ONNXExporter:
    def __init__(self, checkpoint_path: str, output_path: str, config: Config, vocab_size: int | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path)
        self.config = config
        self.vocab_size = vocab_size or config.vocab_size

    def load_model(self) -> GPT:
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
        vocab_size = checkpoint.get("embedding.token_embeddings.weight", checkpoint.get("output_projection.weight")).shape[0] if "embedding.token_embeddings.weight" in checkpoint or "output_projection.weight" in checkpoint else self.vocab_size

        model = GPT(
            vocab_size=vocab_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_context=self.config.max_context,
            feedforward_dim=self.config.feedforward_dim,
            dropout=self.config.dropout,
        )

        clean_state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(clean_state)
        model.eval()

        return model
    
    def export(self) -> None:
        model = self.load_model()

        wrapper = ONNXGPTWrapper(
            model=model,
            num_layers=self.config.num_layers
        )
        head_dim = self.config.embed_dim // self.config.num_heads

        example_tokens = torch.randint(
            low=0,
            high=model.embedding.token_embeddings.num_embeddings,
            size=(1, self.config.max_context),
            dtype=torch.long
        )
        
        example_cache = []

        for _ in range(self.config.num_layers):
            example_cache.extend(
                [
                    torch.zeros(1, self.config.num_heads, 1, head_dim), 
                    torch.zeros(1, self.config.num_heads, 1, head_dim)
                ]
            )

        input_names = ["tokens"]

        for layer in range(self.config.num_layers):
            input_names.extend([
                f"past_key_{layer}",
                f"past_value_{layer}"
            ])
        
        output_names = ["logits"]

        for layer in range(self.config.num_layers):
            output_names.extend([
                f"present_key_{layer}",
                f"present_value_{layer}"
            ])

        batch_dim = torch.export.Dim("batch", min=1)
        seq_dim = torch.export.Dim("sequence", min=1, max=self.config.max_context)
        cache_dim = torch.export.Dim("cache", min=1, max=self.config.max_context)

        dynamic_shapes = {"tokens": {0: batch_dim, 1: seq_dim}}

        for layer in range(self.config.num_layers):
            dynamic_shapes[f"past_key_{layer}"] = {
                0: batch_dim,
                2: cache_dim
            }

            dynamic_shapes[f"past_value_{layer}"] = {
                0: batch_dim,
                2: cache_dim
            }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        onnx_program = torch.onnx.export(
            model=wrapper, 
            args=(example_tokens, *example_cache), 
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
            dynamic_shapes=dynamic_shapes
        )

        onnx_program.save(
            str(self.output_path)
        )


def main() -> None:
    config = Config()

    checkpoint_file = f"tinyGPT_v{config.version}.pth" if Path(f"tinyGPT_v{config.version}.pth").exists() else "artifacts/models/tinygpt-v0.0.1.pth"
    output_file = f"artifacts/models/tinyGPT_v{config.version}.onnx"

    exporter = ONNXExporter(
        checkpoint_path=checkpoint_file,
        output_path=output_file,
        config=config
    )

    exporter.export()

    print(
        f"Exported ONNX model to {exporter.output_path}"
    )

if __name__ == "__main__":
    main()
