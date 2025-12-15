from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .chunking_treesitter import CodeChunk

@dataclass(frozen=True)
class RepoMapConfig:
    max_dirs: int = 80
    max_symbols_per_dir: int = 40

def build_repo_map(chunks: Iterable[CodeChunk], cfg: RepoMapConfig) -> str:
    """
    대규모 레포에서 LLM에게 "길"을 주는 텍스트 맵.
    - 디렉토리별 대표 심볼/시그니처(조각) 나열
    - 심볼 이름 없는 경우도 일부 포함(경계 파악에 도움)
    """
    by_dir: dict[str, list[CodeChunk]] = defaultdict(list)
    for c in chunks:
        d = str(Path(c.rel_path).parent)
        by_dir[d].append(c)

    # 디렉토리 우선순위: 심볼 수 많은 순
    dirs = sorted(by_dir.keys(), key=lambda k: len(by_dir[k]), reverse=True)[:cfg.max_dirs]

    lines: list[str] = []
    lines.append("REPO MAP (directory -> key symbols / boundaries hints)\n")
    for d in dirs:
        lines.append(f"[{d}] (chunks={len(by_dir[d])})")
        # 심볼 우선: 이름 있는 것 + 함수/클래스 계열
        items = by_dir[d]
        items_sorted = sorted(
            items,
            key=lambda c: (
                0 if c.symbol_name else 1,
                0 if c.symbol_kind in ("function_definition", "class_specifier", "struct_specifier", "namespace_definition") else 1,
                -(c.end_line - c.start_line),
            )
        )
        count = 0
        for c in items_sorted:
            if count >= cfg.max_symbols_per_dir:
                break
            name = c.symbol_name or "(anonymous)"
            loc = f"{c.rel_path}:{c.start_line}-{c.end_line}"
            lines.append(f"  - {c.symbol_kind} {name} @ {loc}")
            count += 1
        lines.append("")
    return "\n".join(lines).strip()
