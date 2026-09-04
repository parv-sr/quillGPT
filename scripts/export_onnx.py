import sys
from pathlib import Path
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch import nn
from torch.onnx import register_custom_op_symbolic

from config import Config
from model.gpt import GPT


# Register symbolic handler for aten::rms_norm for ONNX opset compatibility
def _symbolic_rms_norm(g, input, normalized_shape, weight, eps):
    pow_input = g.op("Pow", input, g.op("Constant", value_t=torch.tensor(2.0)))
    mean_input = g.op("ReduceMean", pow_input, axes_i=[-1], keepdims_i=1)
    if eps is None:
        eps = g.op("Constant", value_t=torch.tensor(1e-6))
    add_eps = g.op("Add", mean_input, eps)
    rsqrt = g.op("Sqrt", add_eps)
    div_input = g.op("Div", input, rsqrt)
    if weight is not None:
        return g.op("Mul", div_input, weight)
    return div_input


register_custom_op_symbolic("aten::rms_norm", _symbolic_rms_norm, opset_version=18)


class ONNXGPTWrapper(nn.Module):
    def __init__(self, model: GPT, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(
        self, tokens: torch.Tensor, *past_key_values: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        cache = tuple(
            (past_key_values[i], past_key_values[i + 1])
            for i in range(0, len(past_key_values), 2)
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
    def __init__(
        self,
        checkpoint_path: str,
        output_path: str,
        config: Config,
        vocab_size: int | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path)
        self.config = config
        self.vocab_size = vocab_size or config.vocab_size

    def load_model(self) -> GPT:
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )
        clean_state = {
            k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()
        }

        vocab_size = clean_state["embedding.token_embeddings.weight"].shape[0]
        embed_dim = clean_state["embedding.token_embeddings.weight"].shape[1]

        num_layers = getattr(self.config, "num_layers", 12)
        num_heads = getattr(self.config, "num_heads", 12)
        feedforward_dim = getattr(
            self.config,
            "feedforward_dim",
            clean_state[
                "transformer.blocks.0.feed_forward.gate_projection.weight"
            ].shape[0],
        )
        max_context = getattr(self.config, "max_context", 512)
        head_dim = embed_dim // num_heads

        model = GPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_context=max_context,
            feedforward_dim=feedforward_dim,
            dropout=0.0,
        )

        model.load_state_dict(clean_state, strict=False)
        model.eval()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.max_context = max_context
        self.head_dim = head_dim

        return model

    def export(self) -> None:
        model = self.load_model()

        wrapper = ONNXGPTWrapper(
            model=model, num_layers=self.num_layers
        )
        wrapper.eval()

        example_tokens = torch.randint(
            low=0,
            high=model.embedding.token_embeddings.num_embeddings,
            size=(1, 4),
            dtype=torch.long,
        )

        example_cache = []
        for _ in range(self.num_layers):
            example_cache.extend(
                [
                    torch.zeros(1, self.num_heads, 0, self.head_dim),
                    torch.zeros(1, self.num_heads, 0, self.head_dim),
                ]
            )

        input_names = ["tokens"]
        for layer in range(self.num_layers):
            input_names.extend(
                [f"past_key_{layer}", f"past_value_{layer}"]
            )

        output_names = ["logits"]
        for layer in range(self.num_layers):
            output_names.extend(
                [f"present_key_{layer}", f"present_value_{layer}"]
            )

        # Dynamic dimension definitions for torch.export
        batch_dim = torch.export.Dim("batch", min=1)
        seq_dim = torch.export.Dim("sequence", min=1, max=self.max_context)
        cache_dim = torch.export.Dim("cache", min=0, max=self.max_context)

        tokens_dynamic = {0: batch_dim, 1: seq_dim}
        cache_dynamic = {0: batch_dim, 2: cache_dim}
        
        # Match expected 2-element structure: (tokens_shape, tuple_of_cache_shapes)
        dynamic_shapes = (
            tokens_dynamic,
            tuple(cache_dynamic for _ in example_cache)
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        onnx_program = torch.onnx.export(
            wrapper,
            (example_tokens, *example_cache),
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            dynamo=True,
        )

        onnx_program.save(str(self.output_path))


def main() -> None:
    config = Config()

    checkpoint_file = (
        "artifacts/models/quillGPT_v0.2.3.pth"
        if Path("artifacts/models/quillGPT_v0.2.3.pth").exists()
        else (
            f"quillGPT_v{config.version}.pth"
            if Path(f"quillGPT_v{config.version}.pth").exists()
            else "artifacts/models/tinygpt-v0.0.1.pth"
        )
    )
    output_file = f"artifacts/models/quillGPT_v{config.version}.onnx"

    exporter = ONNXExporter(
        checkpoint_path=checkpoint_file,
        output_path=output_file,
        config=config,
    )

    exporter.export()

    print(f"Exported ONNX model to {exporter.output_path}")


if __name__ == "__main__":
    main()