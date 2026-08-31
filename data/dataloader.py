import torch
from torch.utils.data import DataLoader
from .dataset import LanguageModelDataset


class LanguageModelDataLoader:
    def __init__(
        self,
        train_dataset: LanguageModelDataset,
        validation_dataset: LanguageModelDataset,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        use_pin = pin_memory and torch.cuda.is_available()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=(num_workers > 0),
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=(num_workers > 0),
        )