from pathlib import Path

import torch

from config import Config
from model.gpt import GPT

class ONNXExporter:
    def __init__(self, checkpoint_path: str, output_path: str, config: Config) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path)

        self.config = config

    def load_model(self) -> GPT:
        model = GPT(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_context=self.config.max_context,
            dropout=self.config.dropout,
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(checkpoint)

        model.eval()

        return model
    
    def export(self) ->None:
        model = self.load_model()

        example_tokens = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(1, self.config.max_context),
            dtype=torch.long
        )
        
        batch_dim = torch.export.Dim("batch", min=1)
        seq_dim = torch.export.Dim("sequence", min=1, max=self.config.max_context)
        dynamic_shapes = {"tokens" : {0: batch_dim, 1: seq_dim}}

        onnx = torch.onnx.export(
            model=model, 
            args=(example_tokens), 
            input_names=["tokens"],
            output_names=["logits"],
            dynamo=True,
            dynamic_shapes=dynamic_shapes
        )

        onnx.save(
            self.output_path
        )


def main() -> None:
    config = Config()

    exporter = ONNXExporter(
        checkpoint_path="artifacts/models/tinygpt-v0.0.1.pth",
        output_path="artifacts/models/tinygpt-v0.0.1.onnx",
        config=config
    )

    exporter.export()

    print(
        F"exported ONNX Model to "
        F"{exporter.output_path}"
    )

if __name__ == "__main__":
    main()
