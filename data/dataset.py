from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, context_length: int) -> None:
        super().__init__()
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids, dtype=np.uint16)
        elif token_ids.dtype != np.uint16:
            token_ids = token_ids.astype(np.uint16, copy=False)

        self.tokens: np.ndarray = token_ids
        self.context_length: int = context_length
        self.num_samples: int = (len(self.tokens) - 1) // self.context_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start: int = index * self.context_length
        chunk: np.ndarray = self.tokens[
            start : start + self.context_length + 1
        ]

        input_tokens: torch.Tensor = torch.from_numpy(
            chunk[:-1].astype(np.int64)
        )
        target_tokens: torch.Tensor = torch.from_numpy(
            chunk[1:].astype(np.int64)
        )

        return input_tokens, target_tokens


def train_validation_split(
    token_ids: np.ndarray, validation_fraction: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    split_index: int = int(len(token_ids) * (1 - validation_fraction))
    train_tokens: np.ndarray = token_ids[:split_index]
    validation_tokens: np.ndarray = token_ids[split_index:]

    return train_tokens, validation_tokens