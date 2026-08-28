from pathlib import Path

class TextCorpus:
    def __init__(self, path: Path | str) -> None:
        self.path: Path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Corpus not found at {self.path}")
        
        self.text: str = self._load()

    def _load(self) -> str:
        if self.path.is_dir():
            files = sorted(self.path.glob("input_*.txt"), key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem and p.stem.split("_")[1].isdigit() else p.name)
        
            if not files:
                files = sorted(self.path.glob("*.txt"))
        
            return "".join(f.read_text(encoding="utf-8") for f in files)
        
        return self.path.read_text(encoding="utf-8")
    
    def __len__(self) -> int:
        return len(self.text)

    
