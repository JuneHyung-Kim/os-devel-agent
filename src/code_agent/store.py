from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import json
import pickle

from .chunking_treesitter import CodeChunk, read_chunks_jsonl, write_chunks_jsonl

@dataclass
class IndexMeta:
    repo_root: str
    file_hashes: Dict[str, str]  # rel_path -> sha1
    chunk_count: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: Path) -> "IndexMeta":
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return IndexMeta(**d)

@dataclass
class SymbolIndex:
    # symbol_name_lower -> list of chunk indices
    mapping: Dict[str, List[int]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "SymbolIndex":
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, SymbolIndex):
            raise TypeError("Invalid symbol index")
        return obj

def build_symbol_index(chunks: list[CodeChunk]) -> SymbolIndex:
    m: Dict[str, List[int]] = {}
    for i, c in enumerate(chunks):
        key = (c.symbol_name or "").strip().lower()
        if not key:
            continue
        m.setdefault(key, []).append(i)
    return SymbolIndex(mapping=m)
