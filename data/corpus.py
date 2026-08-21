"""
This file will obtain and load the plain shakespeare's text.
"""
from pathlib import Path

class TextCorpus:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Corpus not found at{self.path}"
            )
        
        self.text = self._load()

    def _load(self) -> str:
        return self.path.read_text(encoding='utf-8')
    
    def __len__(self) -> int:
        return len(self.text)
    
