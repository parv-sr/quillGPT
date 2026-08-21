"""
Implementation of pytorch's abstract base Dataset.
Only overloads __len__ and __getitem__ methods as per our dataset.
Train/Test split: 80/20.
"""


import torch
from torch.utils.data import Dataset

from typing import List, Tuple

class LanguageModelDataset(Dataset):
    def __init__(self, token_ids: List[int], context_length: int) -> None:
        super().__init__()

        self.tokens = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens = self.tokens[index : index + self.context_length]
        target_tokens = self.tokens[index + 1 : index + self.context_length + 1]

        return input_tokens, target_tokens
    

def train_validation_split(token_ids: List[int], validation_fraction: float = 0.2) -> Tuple[List[int], List[int]]:
    split_index = int(
        len(token_ids) * (1 - validation_fraction)
    )

    train_tokens = token_ids[:split_index]
    validation_tokens = token_ids[split_index:]

    return train_tokens, validation_tokens