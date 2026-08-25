"""
This module will prepare shuffled batches of input tensors from the dataset module.
Dataset produces: (T, C)
Dataloader produces: (B, T, C)

The GPUDataloader helper class is wrapped by the main dataloader to optimise training on a cuda device.
"""

from torch.utils.data import DataLoader
from .dataset import LanguageModelDataset
import torch

class GPUDataLoader:
    def __init__(self, dataset: LanguageModelDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

        self.context_length = dataset.context_length
        self.tokens = dataset.tokens

        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // batch_size

    def __len__(self) -> int:
        return self.num_batches
    
    def __iter__(self):
        """
        Works exactly as shuffle=True
        Generates random start positions for every batch in the epoch.
        """
        indices = torch.randperm(self.num_samples, device=self.tokens.device)

        for i in range(self.num_batches):
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]

            offsets = torch.arange(self.context_length, device=self.tokens.device)
            grid_indices = batch_indices.unsqueeze(1) + offsets

            x = self.tokens[grid_indices]
            y = self.tokens[grid_indices + 1]

            yield x, y


class LanguageModelDataLoader:
    def __init__(self, train_dataset: LanguageModelDataset, validation_dataset: LanguageModelDataset, batch_size: int, num_workers: int = 0) -> None:
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers
        ) if not torch.cuda.is_available() else GPUDataLoader(
            train_dataset, batch_size
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers            
        ) if not torch.cuda.is_available() else GPUDataLoader(
            validation_dataset, batch_size
        )
    