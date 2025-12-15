from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import re
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from .base import Retriever, RetrievedChunk
from ..chunking_treesitter import CodeChunk

_token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|->|::|[{}()\[\];,]")

def tokenize_code(text: str) -> list[str]:
    return [t.lower() for t in _token_re.findall(text)]

@dataclass
class BM25Store:
    tokenized_corpus: list[list[str]]
    chunks: list[CodeChunk]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Store":
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, BM25Store):
            raise TypeError("Invalid BM25 store")
        return obj

@dataclass
class BM25Retriever(Retriever):
    store: BM25Store

    def __post_init__(self) -> None:
        self._bm25 = BM25Okapi(self.store.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 8) -> Sequence[RetrievedChunk]:
        q = tokenize_code(query)
        scores = self._bm25.get_scores(q)

        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for i in top_idx:
            c = self.store.chunks[i]
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id,
                rel_path=c.rel_path,
                start_line=c.start_line,
                end_line=c.end_line,
                text=c.text,
                score=float(scores[i]),
                reason="bm25",
            ))
        return out
