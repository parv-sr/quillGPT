import os, re
from pathlib import Path

def rename_raw_files(raw_dir: str = "data/raw"):
    raw_path = Path(raw_dir)
    
    # Ensure 10.txt is formatted if not already
    file_10 = raw_path / "312.txt"
    if file_10.exists():
        file_10.rename(raw_path / "input_312.txt")
        
    # Get all remaining numeric files
    numeric_files = [f for f in raw_path.iterdir() if f.is_file() and re.match(r"^\d+\.txt$", f.name)]
    numeric_files.sort(key=lambda x: int(x.stem))
    
    # Rename starting from 11
    for idx, file_path in enumerate(numeric_files, start=313):
        file_path.rename(raw_path / f"input_{idx}.txt")

if __name__ == "__main__":
    rename_raw_files()