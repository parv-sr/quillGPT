import torch
from pathlib import Path
from torch import nn
from typing import List, Any
from tqdm import tqdm

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from config import Config
from model.gpt import GPT
from data.corpus import TextCorpus
from data.bpe_tokenizer import BPETokenizer
from data.dataset import LanguageModelDataset, train_validation_split
from data.dataloader import LanguageModelDataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        validation_loader: Any,
        config: Config,
        vocab_size: int
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: Any = train_loader
        self.validation_loader: Any = validation_loader

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.use_amp: bool = self.device.type == "cuda"
        self.scaler: torch.amp.GradScaler | None = torch.amp.GradScaler("cuda") if self.use_amp else None

        if self.device.type == "cuda":
            self.model = torch.compile(self.model)

        self.loss_function: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            fused=torch.cuda.is_available()
        )

        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

        self.vocab_size: int = vocab_size

    def train_epoch(self, epoch: int, val_loss: float = 0.0) -> float:
        self.model.train()
        total_loss: float = 0.0
        step_count: int = 0

        pbar: tqdm = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", dynamic_ncols=True, leave=True)

        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                logits: torch.Tensor = self.model(x)
                loss: torch.Tensor = self.loss_function(logits.view(-1, self.vocab_size), y.view(-1))

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            step_count += 1

            current_loss: float = total_loss / step_count
            pbar.set_postfix_str(f"Training Loss={current_loss:.4f}, Validation={val_loss:.4f}", refresh=False)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss: float = 0.0

        for x, y in self.validation_loader:
            if not torch.cuda.is_available():
                x = x.to(self.device)
                y = y.to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                logits: torch.Tensor = self.model(x)
                loss: torch.Tensor = self.loss_function(logits.view(-1, self.vocab_size), y.view(-1))

            total_loss += loss.item()

        return total_loss / len(self.validation_loader)


def main() -> None:
    config: Config = Config()

    corpus: TextCorpus = TextCorpus("data/raw")

    tokenizer: BPETokenizer = BPETokenizer(config.vocab_size)
    tokenizer_path: str = "bpe_tokenizer.json"

    if Path(tokenizer_path).exists():
        tokenizer.load(tokenizer_path)
    else:
        tokenizer.train(corpus.text[:5000000] if len(corpus.text) > 5000000 else corpus.text)
        tokenizer.save(tokenizer_path)

    tokens: List[int] = tokenizer.encode(corpus.text)

    train_tokens, validation_tokens = train_validation_split(tokens)

    train_dataset: LanguageModelDataset = LanguageModelDataset(train_tokens, config.max_context)
    validation_dataset: LanguageModelDataset = LanguageModelDataset(validation_tokens, config.max_context)

    data: LanguageModelDataLoader = LanguageModelDataLoader(train_dataset, validation_dataset, config.batch_size)

    model: GPT = GPT(
        config.vocab_size,
        config.embed_dim,
        config.num_heads,
        config.num_layers,
        config.max_context,
        config.feedforward_dim,
        config.dropout
    )

    trainer: Trainer = Trainer(
        model,
        data.train_loader,
        data.validation_loader,
        config,
        config.vocab_size
    )

    val_loss: float = 0.0

    for epoch in range(config.epochs):
        print("Training loop started")
        train_loss: float = trainer.train_epoch(epoch, val_loss)
        val_loss = trainer.validate()
        trainer.scheduler.step()

    torch.save(model.state_dict(), f"quillGPT_v{config.version}.pth")


if __name__ == "__main__":
    main()