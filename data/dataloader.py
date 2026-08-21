"""
This module will prepare shuffled batches of input tensors from the dataset module.
Dataset produces: (T, C)
Dataloader produces: (B, T, C)
"""

from torch.utils.data import DataLoader
from .dataset import LanguageModelDataset

class LanguageModelDataLoader:
    def __init__(self, train_dataset: LanguageModelDataset, validation_dataset: LanguageModelDataset, batch_size: int, num_workers: int = 0) -> None:
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers            
        )
