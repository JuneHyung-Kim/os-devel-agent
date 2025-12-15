from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from .retrievers.base import RetrievedChunk

@dataclass(frozen=True)
class PackConfig:
    max_chars: int = 80_000
    per_chunk_header: bool = True

def pack_context(repo_map: str, retrieved: Sequence[RetrievedChunk], cfg: PackConfig) -> str:
    parts: list[str] = []
    used = 0

    if repo_map:
        rm = repo_map.strip()
        if len(rm) > cfg.max_chars // 2:
            rm = rm[: cfg.max_chars // 2]
        parts.append(rm + "\n\n")
        used += len(parts[-1])

    for r in retrieved:
        header = ""
        if cfg.per_chunk_header:
            header = f"=== {r.rel_path}:{r.start_line}-{r.end_line} (score={r.score:.3f}) ===\n"
        block = header + r.text.strip() + "\n\n"
        if used + len(block) > cfg.max_chars:
            break
        parts.append(block)
        used += len(block)

    return "".join(parts).strip()
