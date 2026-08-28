import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional

from config import Config
from model.gpt import GPT
from data.corpus import TextCorpus
from data.tokenizer import CharacterTokenizer
from data.bpe_tokenizer import BPETokenizer


class Generator:
    def __init__(self, model: GPT, tokenizer: CharacterTokenizer | BPETokenizer, max_context: int, device: torch.device) -> None:
        self.model: GPT = model
        self.tokenizer: CharacterTokenizer | BPETokenizer = tokenizer
        self.max_context: int = max_context
        self.device: torch.device = device

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 40) -> str:
        token_ids: List[int] = self.tokenizer.encode(prompt)

        if not token_ids:
            return ""

        if len(token_ids) > self.max_context:
            token_ids = token_ids[-self.max_context:]

        tokens: torch.Tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        logits, past_key_values = self.model(tokens, use_cache=True)
        next_logits: torch.Tensor = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")

        probs: torch.Tensor = F.softmax(next_logits, dim=-1)
        next_token: torch.Tensor = torch.multinomial(probs, num_samples=1)

        generated_ids: List[int] = token_ids + [int(next_token.item())]

        for _ in range(max_new_tokens - 1):
            logits, past_key_values = self.model(next_token, past_key_values=past_key_values, use_cache=True)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids.append(int(next_token.item()))

            if len(generated_ids) >= self.max_context:
                break

        return self.tokenizer.decode(generated_ids)


def main() -> None:
    config: Config = Config()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path: Path = Path(f"tinyGPT_v{config.version}.pth")

    if not checkpoint_path.exists():
        checkpoint_path = Path("artifacts/models/tinyGPT_v0.0.2.pth")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}

    vocab_size: int = clean_state["output_projection.weight"].shape[0] if "output_projection.weight" in clean_state else config.vocab_size

    tokenizer: CharacterTokenizer | BPETokenizer

    if Path("bpe_tokenizer.json").exists() and vocab_size > 256:
        bpe = BPETokenizer(vocab_size)
        bpe.load("bpe_tokenizer.json")
        tokenizer = bpe
    else:
        corpus = TextCorpus("data/raw")
        tokenizer = CharacterTokenizer(corpus.text)

    model: GPT = GPT(
        vocab_size,
        config.embed_dim,
        config.num_heads,
        config.num_layers,
        config.max_context,
        config.feedforward_dim,
        config.dropout
    )

    model.load_state_dict(clean_state)

    generator: Generator = Generator(model, tokenizer, config.max_context, device)

    prompt: str = "QUEEN:"
    output: str = generator.generate(prompt=prompt, max_new_tokens=200, temperature=0.8, top_k=40)

    print(output)


if __name__ == "__main__":
    main()