from __future__ import annotations

import argparse
from pathlib import Path

from src.documents import SUPPORTED_EXTENSIONS
from src.vector_store import KnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="Index company documents")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    kb = KnowledgeBase()
    total = 0
    for path in args.paths:
        files = [path] if path.is_file() else list(path.rglob("*"))
        for file in files:
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                count = kb.ingest(file)
                total += count
                print(f"Indexed {file.name}: {count} chunks")
    print(f"Done: {total} chunks indexed")


if __name__ == "__main__":
    main()
