from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import re

from .base import Retriever, RetrievedChunk
from ..chunking_treesitter import CodeChunk
from ..store import SymbolIndex

_word_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

def extract_query_symbols(query: str) -> list[str]:
    # 간단: 식별자 후보 추출
    toks = _word_re.findall(query)
    # 너무 짧은 토큰 제외
    toks = [t for t in toks if len(t) >= 3]
    # 대소문자 혼재 심볼을 고려해 lower 매칭
    return list(dict.fromkeys([t.lower() for t in toks]))

@dataclass
class SymbolRetriever(Retriever):
    chunks: list[CodeChunk]
    sym_index: SymbolIndex

    def retrieve(self, query: str, top_k: int = 8) -> Sequence[RetrievedChunk]:
        candidates: dict[int, float] = {}
        symbols = extract_query_symbols(query)

        for s in symbols:
            hit = self.sym_index.mapping.get(s)
            if not hit:
                continue
            # 심볼명 정확 매칭은 높은 가중치
            for idx in hit:
                candidates[idx] = max(candidates.get(idx, 0.0), 5.0)

        # 부분 매칭(약하게): query token이 심볼명에 포함되는 경우
        if not candidates and symbols:
            for i, c in enumerate(self.chunks):
                name = (c.symbol_name or "").lower()
                if not name:
                    continue
                for s in symbols:
                    if s in name:
                        candidates[i] = max(candidates.get(i, 0.0), 2.0)

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for idx, sc in ranked:
            c = self.chunks[idx]
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id,
                rel_path=c.rel_path,
                start_line=c.start_line,
                end_line=c.end_line,
                text=c.text,
                score=float(sc),
                reason="symbol",
            ))
        return out
