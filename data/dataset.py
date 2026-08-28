import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class LanguageModelDataset(Dataset):
    def __init__(self, token_ids: List[int], context_length: int) -> None:
        super().__init__()

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokens: torch.Tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        self.context_length: int = context_length

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.context_length
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start: int = index * self.context_length

        input_tokens: torch.Tensor = self.tokens[start : start + self.context_length]
        target_tokens: torch.Tensor = self.tokens[start + 1 : start + self.context_length + 1]
        
        return input_tokens, target_tokens

def train_validation_split(token_ids: List[int], validation_fraction: float = 0.2) -> Tuple[List[int], List[int]]:
    split_index: int = int(len(token_ids) * (1 - validation_fraction))
    train_tokens: List[int] = token_ids[:split_index]
    validation_tokens: List[int] = token_ids[split_index:]
    
    return train_tokens, validation_tokens