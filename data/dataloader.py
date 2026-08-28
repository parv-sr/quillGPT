import torch
from torch.utils.data import DataLoader
from typing import Iterator, Tuple
from .dataset import LanguageModelDataset

class GPUDataLoader:
    def __init__(self, dataset: LanguageModelDataset, batch_size: int) -> None:
        self.dataset: LanguageModelDataset = dataset
        self.batch_size: int = batch_size
        self.context_length: int = dataset.context_length
        self.tokens: torch.Tensor = dataset.tokens
        self.num_samples: int = len(dataset)
        self.num_batches: int = self.num_samples // batch_size

    def __len__(self) -> int:
        return self.num_batches
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        indices: torch.Tensor = torch.randperm(self.num_samples, device=self.tokens.device) * self.context_length

        for i in range(self.num_batches):
            batch_indices: torch.Tensor = indices[i * self.batch_size : (i + 1) * self.batch_size]

            offsets: torch.Tensor = torch.arange(self.context_length, device=self.tokens.device)
            
            grid_indices: torch.Tensor = batch_indices.unsqueeze(1) + offsets
            x: torch.Tensor = self.tokens[grid_indices]
            y: torch.Tensor = self.tokens[grid_indices + 1]

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
    