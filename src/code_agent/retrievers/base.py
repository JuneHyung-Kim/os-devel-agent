from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    rel_path: str
    start_line: int
    end_line: int
    text: str
    score: float
    reason: str  # "symbol" / "bm25" / "hybrid"

class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 8) -> Sequence[RetrievedChunk]:
        ...
