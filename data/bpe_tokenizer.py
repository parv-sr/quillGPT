import json
from pathlib import Path
from typing import Dict, List, Tuple
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPETokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size: int = vocab_size
        self._init_empty_tokenizer()

    def _init_empty_tokenizer(self) -> None:
        """Initializes a raw Hugging Face BPE Tokenizer without pre-tokenization rules

        to match character-level exact merging behaviour.
        """
        self.tokenizer = Tokenizer(models.BPE(unk_token=None))
        # No pre_tokenizer is added to match the character-level list(text) behavior.

    def train(self, text: str) -> None:
        """Trains a BPE model using Hugging Face's BpeTrainer."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            show_progress=True,
            initial_alphabet=sorted(list(set(text))),
        )
        # Hugging Face trainers expect iterators over strings/files
        self.tokenizer.train_from_iterator([text], trainer=trainer)

    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of token IDs."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to a string."""
        return self.tokenizer.decode(token_ids)

    def save(self, path: Path | str) -> None:
        """Saves the Hugging Face tokenizer directly to JSON."""
        self.tokenizer.save(str(path))

    def load(self, path: Path | str) -> None:
        """Loads a tokenizer from disk.

        Handles both standard Hugging Face JSON outputs and legacy custom bpe_tokenizer.json format.
        """
        path_str = str(path)
        
        try:
            # Try loading as a standard Hugging Face Tokenizer JSON first
            self.tokenizer = Tokenizer.from_file(path_str)
            self.vocab_size = self.tokenizer.get_vocab_size()
        except Exception:
            # Fallback: Convert custom format to HF BPE model directly
            with open(path_str, "r", encoding="utf-8") as f:
                data = json.load(f)

            vocab: Dict[str, int] = {v: int(k) for k, v in data["vocab"].items()}
            merges: List[Tuple[str, str]] = [(item[0], item[1]) for item in data["merges"]]

            bpe_model = models.BPE(vocab=vocab, merges=merges, unk_token=None)
            self.tokenizer = Tokenizer(bpe_model)
            self.vocab_size = data.get("vocab_size", len(vocab))