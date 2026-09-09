import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config
from data.bpe_tokenizer import BPETokenizer
from data.dataloader import LanguageModelDataLoader
from data.dataset import LanguageModelDataset
from model.gpt import GPT

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

LOG_INTERVAL_STEPS = 10
RESUME = os.environ.get("QUILL_RESUME", "0") == "1"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        validation_loader: Any,
        config: Config,
        vocab_size: int,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.vocab_size = vocab_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.raw_model = model.to(self.device)
        self.model = torch.compile(self.raw_model)

        self.use_bf16 = (
            self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        self.amp_dtype = (
            torch.bfloat16
            if self.use_bf16
            else torch.float16
        )
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.device.type == "cuda"
            and not self.use_bf16
            else None
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.gradient_clip = 1.0

        decay_params = [
            p
            for p in self.raw_model.parameters()
            if p.requires_grad and p.dim() >= 2
        ]
        nodecay_params = [
            p
            for p in self.raw_model.parameters()
            if p.requires_grad and p.dim() < 2
        ]

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": nodecay_params,
                    "weight_decay": 0.0,
                },
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            fused=(self.device.type == "cuda"),
        )

        self.tokens_per_microbatch = (
            config.batch_size
            * config.max_context
        )
        self.tokens_per_optimizer_step = (
            self.tokens_per_microbatch
            * config.gradient_accumulation_steps
        )
        self.total_steps = math.ceil(
            config.max_train_tokens
            / self.tokens_per_optimizer_step
        )
        self.warmup_steps = max(
            1,
            math.ceil(
                config.warmup_tokens
                / self.tokens_per_optimizer_step
            ),
        )

        minimum_ratio = (
            config.min_learning_rate
            / config.learning_rate
        )

        def learning_rate_lambda(
            step: int,
        ) -> float:
            if step < self.warmup_steps:
                return (
                    float(step + 1)
                    / self.warmup_steps
                )

            progress = (
                step - self.warmup_steps
            ) / max(
                1,
                self.total_steps
                - self.warmup_steps,
            )
            progress = min(
                max(progress, 0.0),
                1.0,
            )
            cosine = 0.5 * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )

            return (
                minimum_ratio
                + (1.0 - minimum_ratio)
                * cosine
            )

        self.scheduler = (
            torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=learning_rate_lambda,
            )
        )

        self.tokens_seen = 0
        self.optimizer_step = 0
        self.sample_offset = 0
        self.pass_index = 0
        self.best_validation_loss = float("inf")
        self.last_validation_loss = float("nan")

        self.checkpoint_dir = Path(
            "checkpoints"
        )
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.latest_checkpoint_path = (
            self.checkpoint_dir
            / f"quillGPT_v{config.version}_latest.pt"
        )
        self.best_model_path = Path(
            f"quillGPT_v{config.version}_best.pth"
        )
        self.final_model_path = Path(
            f"quillGPT_v{config.version}.pth"
        )

        if RESUME:
            self.load_checkpoint(
                self.latest_checkpoint_path
            )

    def _atomic_save(self, obj: Any, path: Path) -> None:
        tmp_path = path.with_name(
            path.name + ".tmp"
        )
        torch.save(
            obj,
            tmp_path,
        )
        os.replace(
            tmp_path,
            path,
        )

    def _checkpoint_state(self) -> dict[str, Any]:
        state = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": (
                self.scaler.state_dict()
                if self.scaler is not None
                else None
            ),
            "tokens_seen": self.tokens_seen,
            "optimizer_step": self.optimizer_step,
            "sample_offset": self.sample_offset,
            "pass_index": self.pass_index,
            "best_validation_loss": (
                self.best_validation_loss
            ),
            "last_validation_loss": (
                self.last_validation_loss
            ),
            "torch_rng_state": (
                torch.get_rng_state()
            ),
            "version": self.config.version,
            "vocab_size": self.vocab_size,
            "max_context": (
                self.config.max_context
            ),
        }

        if self.device.type == "cuda":
            state[
                "cuda_rng_state_all"
            ] = torch.cuda.get_rng_state_all()

        return state

    def save_checkpoint(
        self,
    ) -> None:
        self._atomic_save(
            self._checkpoint_state(),
            self.latest_checkpoint_path,
        )

    def save_best_model(
        self,
    ) -> None:
        self._atomic_save(
            self.raw_model.state_dict(),
            self.best_model_path,
        )

    def save_final_model(
        self,
    ) -> None:
        self._atomic_save(
            self.raw_model.state_dict(),
            self.final_model_path,
        )

    def load_checkpoint(
        self,
        path: Path,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(
                "Resume requested but "
                f"checkpoint does not exist: {path}"
            )

        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                path,
                map_location=self.device,
            )

        if (
            checkpoint.get("vocab_size")
            != self.vocab_size
        ):
            raise ValueError(
                "Checkpoint vocabulary size "
                "does not match the current model."
            )

        if (
            checkpoint.get("max_context")
            != self.config.max_context
        ):
            raise ValueError(
                "Checkpoint context length "
                "does not match the current config."
            )

        self.raw_model.load_state_dict(
            checkpoint["model"]
        )
        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        self.scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        if (
            self.scaler is not None
            and checkpoint.get("scaler")
            is not None
        ):
            self.scaler.load_state_dict(
                checkpoint["scaler"]
            )

        self.tokens_seen = int(
            checkpoint.get(
                "tokens_seen",
                0,
            )
        )
        self.optimizer_step = int(
            checkpoint.get(
                "optimizer_step",
                0,
            )
        )
        self.sample_offset = int(
            checkpoint.get(
                "sample_offset",
                0,
            )
        )
        self.pass_index = int(
            checkpoint.get(
                "pass_index",
                0,
            )
        )
        self.best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                float("inf"),
            )
        )
        self.last_validation_loss = float(
            checkpoint.get(
                "last_validation_loss",
                float("nan"),
            )
        )

        if (
            "torch_rng_state"
            in checkpoint
        ):
            torch.set_rng_state(
                checkpoint[
                    "torch_rng_state"
                ].cpu()
            )

        if (
            self.device.type == "cuda"
            and "cuda_rng_state_all"
            in checkpoint
        ):
            torch.cuda.set_rng_state_all(
                [
                    state.cpu()
                    for state in checkpoint[
                        "cuda_rng_state_all"
                    ]
                ]
            )

        logger.info(
            "Resumed from %s at %s tokens, "
            "optimizer step %s, pass %d",
            path,
            f"{self.tokens_seen:,}",
            f"{self.optimizer_step:,}",
            self.pass_index + 1,
        )

    def _train_loader_from_offset(
        self,
    ) -> DataLoader:
        if self.sample_offset <= 0:
            return self.train_loader

        dataset = self.train_loader.dataset

        if (
            self.sample_offset
            >= len(dataset)
        ):
            self.sample_offset = 0
            self.pass_index += 1
            return self.train_loader

        subset = Subset(
            dataset,
            range(
                self.sample_offset,
                len(dataset),
            ),
        )

        kwargs: dict[str, Any] = {
            "batch_size": (
                self.train_loader.batch_size
            ),
            "shuffle": False,
            "drop_last": True,
            "num_workers": (
                self.train_loader.num_workers
            ),
            "pin_memory": (
                self.train_loader.pin_memory
            ),
            "persistent_workers": (
                self.train_loader
                .persistent_workers
            ),
        }

        if (
            self.train_loader.num_workers
            > 0
        ):
            kwargs[
                "prefetch_factor"
            ] = (
                self.train_loader
                .prefetch_factor
            )

        return DataLoader(
            subset,
            **kwargs,
        )

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()

        total_loss = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float32,
        )
        step_count = 0

        for batch in self.validation_loader:
            tokens = batch.to(
                self.device,
                dtype=torch.long,
                non_blocking=True,
            )

            x = tokens[:, :-1]
            y = tokens[:, 1:]

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=(
                    self.device.type
                    == "cuda"
                ),
            ):
                logits = self.model(x)

                loss = self.loss_function(
                    logits.reshape(
                        -1,
                        self.vocab_size,
                    ),
                    y.reshape(-1),
                )

            total_loss += (
                loss.detach().float()
            )
            step_count += 1

            if (
                step_count
                >= self.config.validation_batches
            ):
                break

        if step_count == 0:
            raise RuntimeError(
                "Validation loader "
                "produced no batches."
            )

        validation_loss = (
            total_loss / step_count
        ).item()

        self.model.train()

        return validation_loss

    def train(
        self,
    ) -> None:
        self.model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        next_validation_token = (
            (
                self.tokens_seen
                // self.config
                .validation_interval_tokens
            )
            + 1
        ) * (
            self.config
            .validation_interval_tokens
        )

        next_checkpoint_token = (
            (
                self.tokens_seen
                // self.config
                .checkpoint_interval_tokens
            )
            + 1
        ) * (
            self.config
            .checkpoint_interval_tokens
        )

        progress_initial = min(
            self.tokens_seen,
            self.config.max_train_tokens,
        )

        pbar = tqdm(
            total=self.config.max_train_tokens,
            initial=progress_initial,
            desc=(
                f"Training Pass "
                f"{self.pass_index + 1}"
            ),
            unit="tok",
            unit_scale=True,
            dynamic_ncols=True,
            leave=True,
            mininterval=1.0,
        )

        running_loss = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float32,
        )
        running_microbatches = 0
        window_tokens = 0
        window_start = (
            time.perf_counter()
        )

        accumulation_count = 0

        accumulation_start_tokens = (
            self.tokens_seen
        )
        accumulation_start_offset = (
            self.sample_offset
        )
        accumulation_start_pass = (
            self.pass_index
        )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        reached_target = (
            self.tokens_seen
            >= self.config.max_train_tokens
        )

        try:
            while not reached_target:
                active_loader = (
                    self._train_loader_from_offset()
                )

                for batch in active_loader:
                    if (
                        accumulation_count
                        == 0
                    ):
                        accumulation_start_tokens = (
                            self.tokens_seen
                        )
                        accumulation_start_offset = (
                            self.sample_offset
                        )
                        accumulation_start_pass = (
                            self.pass_index
                        )

                    tokens = batch.to(
                        self.device,
                        dtype=torch.long,
                        non_blocking=True,
                    )

                    x = tokens[:, :-1]
                    y = tokens[:, 1:]
                    batch_tokens = x.numel()

                    with torch.autocast(
                        device_type=(
                            self.device.type
                        ),
                        dtype=self.amp_dtype,
                        enabled=(
                            self.device.type
                            == "cuda"
                        ),
                    ):
                        logits = self.model(x)

                        loss = (
                            self.loss_function(
                                logits.reshape(
                                    -1,
                                    self.vocab_size,
                                ),
                                y.reshape(-1),
                            )
                        )

                        scaled_loss = (
                            loss
                            / self.config
                            .gradient_accumulation_steps
                        )

                    if (
                        self.scaler
                        is not None
                    ):
                        self.scaler.scale(
                            scaled_loss
                        ).backward()
                    else:
                        scaled_loss.backward()

                    running_loss += (
                        loss.detach().float()
                    )
                    running_microbatches += 1
                    window_tokens += (
                        batch_tokens
                    )
                    accumulation_count += 1
                    self.tokens_seen += (
                        batch_tokens
                    )
                    self.sample_offset += (
                        tokens.size(0)
                    )

                    remaining_progress = max(
                        0,
                        self.config.max_train_tokens
                        - pbar.n,
                    )

                    pbar.update(
                        min(
                            batch_tokens,
                            remaining_progress,
                        )
                    )

                    if (
                        accumulation_count
                        < self.config
                        .gradient_accumulation_steps
                    ):
                        continue

                    if (
                        self.scaler
                        is not None
                    ):
                        self.scaler.unscale_(
                            self.optimizer
                        )

                    torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(),
                        self.gradient_clip,
                    )

                    if (
                        self.scaler
                        is not None
                    ):
                        self.scaler.step(
                            self.optimizer
                        )
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                    self.scheduler.step()

                    accumulation_count = 0
                    self.optimizer_step += 1

                    if (
                        self.optimizer_step == 1
                        or self.optimizer_step
                        % LOG_INTERVAL_STEPS
                        == 0
                    ):
                        if (
                            self.device.type
                            == "cuda"
                        ):
                            torch.cuda.synchronize()

                        elapsed = max(
                            time.perf_counter()
                            - window_start,
                            1e-9,
                        )

                        tokens_per_second = (
                            window_tokens
                            / elapsed
                        )

                        current_loss = (
                            running_loss
                            / max(
                                1,
                                running_microbatches,
                            )
                        ).item()

                        perplexity = math.exp(
                            min(
                                current_loss,
                                20.0,
                            )
                        )

                        current_lr = (
                            self.optimizer
                            .param_groups[0]["lr"]
                        )

                        if math.isfinite(
                            self.last_validation_loss
                        ):
                            val_text = (
                                f"{self.last_validation_loss:.4f}"
                            )
                        else:
                            val_text = "n/a"

                        if (
                            self.device.type
                            == "cuda"
                        ):
                            peak_allocated = (
                                torch.cuda
                                .max_memory_allocated()
                                / 1024**3
                            )
                            peak_reserved = (
                                torch.cuda
                                .max_memory_reserved()
                                / 1024**3
                            )
                            memory_text = (
                                f"{peak_allocated:.2f}/"
                                f"{peak_reserved:.2f}G"
                            )
                        else:
                            memory_text = "CPU"

                        pbar.set_postfix_str(
                            f"Loss={current_loss:.4f}, "
                            f"Val={val_text}, "
                            f"PPL={perplexity:.2f}, "
                            f"LR={current_lr:.2e}, "
                            f"Tok/s={tokens_per_second:,.0f}, "
                            f"VRAM={memory_text}, "
                            f"Step={self.optimizer_step:,}",
                            refresh=True,
                        )

                        running_loss.zero_()
                        running_microbatches = 0
                        window_tokens = 0
                        window_start = (
                            time.perf_counter()
                        )

                        if (
                            self.device.type
                            == "cuda"
                        ):
                            torch.cuda.reset_peak_memory_stats()

                    if (
                        self.tokens_seen
                        >= next_validation_token
                    ):
                        if (
                            self.device.type
                            == "cuda"
                        ):
                            torch.cuda.synchronize()

                        validation_loss = (
                            self.validate()
                        )

                        self.last_validation_loss = (
                            validation_loss
                        )

                        validation_perplexity = (
                            math.exp(
                                min(
                                    validation_loss,
                                    20.0,
                                )
                            )
                        )

                        tqdm.write(
                            f"Validation at "
                            f"{self.tokens_seen:,} tokens | "
                            f"Loss={validation_loss:.4f} | "
                            f"PPL={validation_perplexity:.2f}"
                        )

                        if (
                            validation_loss
                            < self.best_validation_loss
                        ):
                            self.best_validation_loss = (
                                validation_loss
                            )

                            self.save_best_model()

                            tqdm.write(
                                f"New best model saved to "
                                f"{self.best_model_path}"
                            )

                        while (
                            next_validation_token
                            <= self.tokens_seen
                        ):
                            next_validation_token += (
                                self.config
                                .validation_interval_tokens
                            )

                        running_loss.zero_()
                        running_microbatches = 0
                        window_tokens = 0
                        window_start = (
                            time.perf_counter()
                        )

                        if (
                            self.device.type
                            == "cuda"
                        ):
                            torch.cuda.reset_peak_memory_stats()

                    if (
                        self.tokens_seen
                        >= next_checkpoint_token
                    ):
                        self.save_checkpoint()

                        tqdm.write(
                            f"Checkpoint saved at "
                            f"{self.tokens_seen:,} tokens to "
                            f"{self.latest_checkpoint_path}"
                        )

                        while (
                            next_checkpoint_token
                            <= self.tokens_seen
                        ):
                            next_checkpoint_token += (
                                self.config
                                .checkpoint_interval_tokens
                            )

                        window_start = (
                            time.perf_counter()
                        )

                        if (
                            self.device.type
                            == "cuda"
                        ):
                            torch.cuda.reset_peak_memory_stats()

                    if (
                        self.tokens_seen
                        >= self.config.max_train_tokens
                    ):
                        reached_target = True
                        break

                if reached_target:
                    break

                self.sample_offset = 0
                self.pass_index += 1

                pbar.set_description(
                    f"Training Pass "
                    f"{self.pass_index + 1}"
                )

            self.save_checkpoint()
            self.save_final_model()

            logger.info(
                "Training complete."
            )
            logger.info(
                "Final model saved to %s",
                self.final_model_path,
            )
            logger.info(
                "Final checkpoint saved to %s",
                self.latest_checkpoint_path,
            )
            logger.info(
                "Tokens seen: %s",
                f"{self.tokens_seen:,}",
            )
            logger.info(
                "Optimizer steps: %s",
                f"{self.optimizer_step:,}",
            )
            logger.info(
                "Best validation loss: %.4f",
                self.best_validation_loss,
            )

        except KeyboardInterrupt:
            if accumulation_count != 0:
                self.tokens_seen = (
                    accumulation_start_tokens
                )
                self.sample_offset = (
                    accumulation_start_offset
                )
                self.pass_index = (
                    accumulation_start_pass
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

            self.save_checkpoint()

            logger.info(
                "Interrupted. "
                "Checkpoint saved to %s",
                self.latest_checkpoint_path,
            )

            raise

        finally:
            pbar.close()


def main() -> None:
    config = Config()

    train_bin_path = Path(
        "data/tokens/train.bin"
    )
    val_bin_path = Path(
        "data/tokens/validation.bin"
    )

    if (
        not train_bin_path.exists()
        or not val_bin_path.exists()
    ):
        logger.info(
            "Token binary files not found. "
            "Running prepare_corpus.py..."
        )

        from prepare_corpus import prepare_corpus

        prepare_corpus()

    tokenizer_path = Path(
        f"bpe_tokenizer_{config.vocab_size}.json"
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: "
            f"{tokenizer_path}"
        )

    tokenizer = BPETokenizer(
        config.vocab_size
    )

    logger.info(
        "Loading tokenizer from %s",
        tokenizer_path,
    )

    tokenizer.load(
        tokenizer_path
    )

    if (
        tokenizer.vocab_size
        != config.vocab_size
    ):
        raise ValueError(
            f"Tokenizer vocabulary size "
            f"{tokenizer.vocab_size} "
            f"does not match "
            f"config.vocab_size="
            f"{config.vocab_size}"
        )

    train_tokens = np.memmap(
        train_bin_path,
        dtype=np.uint16,
        mode="r",
    )

    validation_tokens = np.memmap(
        val_bin_path,
        dtype=np.uint16,
        mode="r",
    )

    logger.info(
        "Training tokens loaded "
        "via memmap: %s",
        f"{len(train_tokens):,}",
    )

    logger.info(
        "Validation tokens loaded "
        "via memmap: %s",
        f"{len(validation_tokens):,}",
    )

    train_dataset = (
        LanguageModelDataset(
            train_tokens,
            config.max_context,
        )
    )

    validation_dataset = (
        LanguageModelDataset(
            validation_tokens,
            config.max_context,
        )
    )

    data = LanguageModelDataLoader(
        train_dataset,
        validation_dataset,
        config.batch_size,
        num_workers=config.num_workers,
    )

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_context=config.max_context,
        feedforward_dim=(
            config.feedforward_dim
        ),
        dropout=config.dropout,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    effective_batch_tokens = (
        config.batch_size
        * config.max_context
        * config.gradient_accumulation_steps
    )

    logger.info(
        "Model parameters: %s",
        f"{parameter_count:,}",
    )

    logger.info(
        "Device: %s",
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
    )

    logger.info(
        "Microbatch size: %d",
        config.batch_size,
    )

    logger.info(
        "Gradient accumulation steps: %d",
        config.gradient_accumulation_steps,
    )

    logger.info(
        "Context length: %d",
        config.max_context,
    )

    logger.info(
        "Effective batch tokens: %s",
        f"{effective_batch_tokens:,}",
    )

    logger.info(
        "Target training tokens: %s",
        f"{config.max_train_tokens:,}",
    )

    logger.info(
        "Warmup tokens: %s",
        f"{config.warmup_tokens:,}",
    )

    logger.info(
        "Validation interval: %s tokens",
        f"{config.validation_interval_tokens:,}",
    )

    logger.info(
        "Checkpoint interval: %s tokens",
        f"{config.checkpoint_interval_tokens:,}",
    )

    trainer = Trainer(
        model,
        data.train_loader,
        data.validation_loader,
        config,
        tokenizer.vocab_size,
    )

    logger.info(
        "Planned optimizer steps: %s",
        f"{trainer.total_steps:,}",
    )

    logger.info(
        "Warmup optimizer steps: %s",
        f"{trainer.warmup_steps:,}",
    )

    logger.info(
        "Precision mode: %s",
        (
            "bfloat16"
            if trainer.use_bf16
            else "float16"
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()