import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def process_single_file(file_info: tuple[Path, Path, Path]) -> None:
    file_path, input_dir, output_dir = file_info
    relative_path = file_path.relative_to(input_dir)
    dest_path = output_dir / relative_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Regex operations
    text = re.sub(r'(?i)Produced by.*?Internet Archive\)', '', text, flags=re.DOTALL)
    text = re.sub(r'(?i)TRANSCRIBER’S NOTES:.*?(?=\n\s*\n\s*[A-Z]|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'(?i)\n\s*CONTENTS\..*?(?=\n\s*(?:CHAPTER|[A-Z\s]{4,})|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'(?i)\n\s*ILLUSTRATIONS\..*?(?=\n\s*(?:CHAPTER|[A-Z\s]{4,})|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'\[Illustration:[^\]]*\]', '', text)
    text = re.sub(r'<DW\d+>', '', text)
    text = re.sub(r'\.{2,}\s*\d+', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text.strip())


if __name__ == "__main__":
    input_dir = Path("data/raw")
    output_dir = Path("data/cleaned")
    
    raw_files = list(input_dir.rglob("*.txt"))
    tasks = [(f, input_dir, output_dir) for f in raw_files]

    # Uses available CPU threads on the remote server
    num_workers = os.cpu_count() or 4
    print(f"Cleaning {len(raw_files)} files using {num_workers} processes...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), 1):
            if idx % 500 == 0:
                print(f"Finished {idx}/{len(raw_files)} files")