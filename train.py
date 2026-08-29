import logging
import math
from pathlib import Path
from typing import Any, List

import torch
from torch import nn
from tqdm import tqdm

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from config import Config
from data.bpe_tokenizer import BPETokenizer
from data.corpus import TextCorpus
from data.dataloader import LanguageModelDataLoader
from data.dataset import LanguageModelDataset, train_validation_split
from model.gpt import GPT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        validation_loader: Any,
        config: Config,
        vocab_size: int,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: Any = train_loader
        self.validation_loader: Any = validation_loader
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Detect Ampere/Ada support for bfloat16
        self.use_bf16: bool = (
            self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        self.amp_dtype: torch.dtype = (
            torch.bfloat16 if self.use_bf16 else torch.float16
        )

        # GradScaler is only necessary for FP16, not BF16
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler("cuda")
            if (self.device.type == "cuda" and not self.use_bf16)
            else None
        )

        self.loss_function: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.vocab_size: int = vocab_size
        self.gradient_clip: float = 1.0

        decay_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]
        nodecay_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.dim() < 2
        ]

        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            fused=(self.device.type == "cuda"),
        )

        self.total_steps: int = len(self.train_loader) * config.epochs
        self.warmup_steps: int = int(self.total_steps * config.warmup_fraction)

        def learning_rate_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step + 1) / max(1, self.warmup_steps)

            progress: float = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine: float = 0.5 * (1.0 + math.cos(progress * math.pi))
            minimum_ratio: float = (
                config.min_learning_rate / config.learning_rate
            )

            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        self.scheduler: torch.optim.lr_scheduler.LambdaLR = (
            torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=learning_rate_lambda
            )
        )

    def train_epoch(self, epoch: int, val_loss: float = 0.0) -> float:
        self.model.train()
        total_loss: float = 0.0
        step_count: int = 0

        pbar: tqdm = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch + 1}",
            dynamic_ncols=True,
            leave=True,
        )

        for x, y in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=True
            ):
                logits: torch.Tensor = self.model(x)
                loss: torch.Tensor = self.loss_function(
                    logits.view(-1, self.vocab_size), y.view(-1)
                )

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            step_count += 1

            current_loss: float = total_loss / step_count
            current_lr: float = self.optimizer.param_groups[0]["lr"]
            perplexity: float = math.exp(min(current_loss, 20.0))

            pbar.set_postfix_str(
                f"Loss={current_loss:.4f}, Val={val_loss:.4f}, PPL={perplexity:.2f}, LR={current_lr:.2e}",
                refresh=False,
            )

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss: float = 0.0
        step_count: int = 0

        for x, y in self.validation_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=True
            ):
                logits: torch.Tensor = self.model(x)
                loss: torch.Tensor = self.loss_function(
                    logits.view(-1, self.vocab_size), y.view(-1)
                )

            total_loss += loss.item()
            step_count += 1

        return total_loss / step_count


def main() -> None:
    config: Config = Config()

    logger.info("Loading corpus...")
    corpus: TextCorpus = TextCorpus("data/cleaned")
    logger.info("Corpus loaded: %d characters", len(corpus.text))

    tokenizer: BPETokenizer = BPETokenizer(config.vocab_size)
    tokenizer_path: str = f"bpe_tokenizer_{config.vocab_size}.json"

    if Path(tokenizer_path).exists():
        logger.info("Loading tokenizer from %s", tokenizer_path)
        tokenizer.load(tokenizer_path)
    else:
        logger.info(
            "Training BPE tokenizer with vocabulary size %d...",
            config.vocab_size,
        )
        tokenizer.train(corpus.text)
        tokenizer.save(tokenizer_path)
        logger.info("Tokenizer saved to %s", tokenizer_path)

    logger.info("Tokenizer vocabulary size: %d", tokenizer.vocab_size)
    tokens: List[int] = tokenizer.encode(corpus.text)
    logger.info("Corpus tokenized: %d tokens", len(tokens))

    train_tokens, validation_tokens = train_validation_split(tokens)
    logger.info("Training tokens: %d", len(train_tokens))
    logger.info("Validation tokens: %d", len(validation_tokens))

    train_dataset: LanguageModelDataset = LanguageModelDataset(
        train_tokens, config.max_context
    )
    validation_dataset: LanguageModelDataset = LanguageModelDataset(
        validation_tokens, config.max_context
    )

    data: LanguageModelDataLoader = LanguageModelDataLoader(
        train_dataset, validation_dataset, config.batch_size
    )

    model: GPT = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_context=config.max_context,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
    )

    parameter_count: int = sum(
        parameter.numel() for parameter in model.parameters()
    )

    logger.info("Model parameters: %s", f"{parameter_count:,}")
    logger.info(
        "Device: %s",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
    )
    logger.info("Batch size: %d", config.batch_size)
    logger.info("Context length: %d", config.max_context)
    logger.info("Training epochs: %d", config.epochs)

    trainer: Trainer = Trainer(
        model,
        data.train_loader,
        data.validation_loader,
        config,
        tokenizer.vocab_size,
    )

    logger.info("Total training steps: %d", trainer.total_steps)
    logger.info("Warmup steps: %d", trainer.warmup_steps)
    logger.info("Precision Mode: %s", "bfloat16" if trainer.use_bf16 else "float16")

    best_validation_loss: float = float("inf")

    for epoch in range(config.epochs):
        logger.info("Starting epoch %d/%d", epoch + 1, config.epochs)

        train_loss: float = trainer.train_epoch(epoch, val_loss=best_validation_loss)
        validation_loss: float = trainer.validate()

        train_perplexity: float = math.exp(min(train_loss, 20.0))
        validation_perplexity: float = math.exp(min(validation_loss, 20.0))
        current_lr: float = trainer.optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Train PPL: %.2f | Val PPL: %.2f | LR: %.2e",
            epoch + 1,
            config.epochs,
            train_loss,
            validation_loss,
            train_perplexity,
            validation_perplexity,
            current_lr,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                model.state_dict(), f"quillGPT_v{config.version}_best.pth"
            )
            logger.info(
                "New best model saved with validation loss %.4f",
                validation_loss,
            )

    torch.save(model.state_dict(), f"quillGPT_v{config.version}.pth")
    logger.info("Training complete.")
    logger.info("Final model saved to quillGPT_v%s.pth", config.version)
    logger.info("Best validation loss: %.4f", best_validation_loss)


if __name__ == "__main__":
    main()