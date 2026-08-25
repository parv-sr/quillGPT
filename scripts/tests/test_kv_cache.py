import torch

from config import Config
from model.gpt import GPT


config = Config()

model = GPT(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    max_context=config.max_context,
    feedforward_dim=config.feedforward_dim,
    dropout=0.0
)

model.eval()

tokens = torch.randint(
    0,
    config.vocab_size,
    (1, 16)
)

with torch.no_grad():

    # Full forward pass
    full_logits = model(tokens)

    # Prefill
    cached_logits, cache = model(
        tokens[:, :-1],
        use_cache=True
    )

    # Decode the final token
    next_logits, cache = model(
        tokens[:, -1:],
        past_key_values=cache,
        use_cache=True
    )

full_last_logits = full_logits[:, -1, :]
cached_last_logits = next_logits[:, -1, :]

difference = (
    full_last_logits - cached_last_logits
).abs()

print("Full logits:", full_last_logits.shape)
print("Cached logits:", cached_last_logits.shape)
print("Maximum difference:", difference.max().item())
print("Mean difference:", difference.mean().item())