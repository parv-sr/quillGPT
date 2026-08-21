import torch
from torch import nn

from typing import Tuple, List

from config import Config
from model.gpt import GPT
from data.corpus import TextCorpus
from data.tokenizer import CharacterTokenizer
from data.dataset import (
    LanguageModelDataset,
    train_validation_split,
)
from data.dataloader import LanguageModelDataLoader

class Trainer:
    def __init__(self, model: nn.Module, train_loader, validation_loader, config: Config) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        self.vocab_size = config.vocab_size

    def train_epoch(self) -> float:
        self.model.train()

        total_loss: float = 0.0

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)

            loss = self.loss_function(
                logits.view(-1, self.vocab_size),
                y.view(-1)
            )

            loss.backward()

            self.optimizer.step()

            total_loss +- loss.item()

        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()

        total_loss: float

        
        for x, y in self.validation_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)

            loss = self.loss_function(
                logits.view(-1, self.vocab_size),
                y.view(-1),
            )

            total_loss += loss.item()

        return total_loss / len(self.validation_loader)
    

def main() -> None:
    config = Config()

    corpus = TextCorpus("data/raw/input.txt")
    tokenizer = CharacterTokenizer(corpus.text)
    tokens = tokenizer.encode(corpus.text)

    train_tokens, validation_tokens = (train_validation_split(tokens))

    train_dataset = LanguageModelDataset(
        train_tokens,
        config.max_context
    )

    validation_dataset = LanguageModelDataset(
        validation_tokens,
        config.max_context
    )

    data = LanguageModelDataLoader(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=config.batch_size
    )

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_context=config.max_context,
        dropout=config.dropout
    )

    trainer = Trainer(
        model=model,
        train_loader=data.train_loader,
        validation_loader=data.validation_loader,
        config=config
    )

    print(f"Device: {trainer.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(config.epochs):
        train_loss = trainer.train_epoch()
        validation_loss = trainer.validate()

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f}"
        )

if __name__ == "__main__":
    main()