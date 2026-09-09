from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        token_ids: np.ndarray,
        context_length: int,
    ) -> None:
        super().__init__()

        if token_ids.dtype != np.uint16:
            raise ValueError(f"Expected uint16 token storage, got {token_ids.dtype}")
        
        self.tokens = token_ids
        self.context_length = context_length

        self.num_samples = (
            len(self.tokens) - 1
        ) // self.context_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.context_length
        end = start + self.context_length + 1

        chunk = np.array(
            self.tokens[start:end],
            dtype=np.uint16,
            copy=True,
        )

        return torch.from_numpy(chunk)