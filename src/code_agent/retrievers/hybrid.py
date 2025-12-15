from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from .base import Retriever, RetrievedChunk

@dataclass
class HybridRetriever(Retriever):
    a: Retriever
    b: Retriever
    weight_a: float = 1.0
    weight_b: float = 1.0

    def retrieve(self, query: str, top_k: int = 8) -> Sequence[RetrievedChunk]:
        ra = self.a.retrieve(query, top_k=top_k * 2)
        rb = self.b.retrieve(query, top_k=top_k * 2)

        # chunk_id 기준 합산
        merged: dict[str, RetrievedChunk] = {}

        def add(items: Sequence[RetrievedChunk], w: float, tag: str) -> None:
            for it in items:
                sc = it.score * w
                if it.chunk_id not in merged:
                    merged[it.chunk_id] = RetrievedChunk(
                        chunk_id=it.chunk_id,
                        rel_path=it.rel_path,
                        start_line=it.start_line,
                        end_line=it.end_line,
                        text=it.text,
                        score=sc,
                        reason=tag,
                    )
                else:
                    prev = merged[it.chunk_id]
                    merged[it.chunk_id] = RetrievedChunk(
                        chunk_id=prev.chunk_id,
                        rel_path=prev.rel_path,
                        start_line=prev.start_line,
                        end_line=prev.end_line,
                        text=prev.text,
                        score=prev.score + sc,
                        reason="hybrid",
                    )

        add(ra, self.weight_a, "a")
        add(rb, self.weight_b, "b")

        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:top_k]
        # reason 정리
        out: list[RetrievedChunk] = []
        for it in ranked:
            out.append(RetrievedChunk(
                chunk_id=it.chunk_id,
                rel_path=it.rel_path,
                start_line=it.start_line,
                end_line=it.end_line,
                text=it.text,
                score=it.score,
                reason="hybrid",
            ))
        return out
