from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from .config import RepoScanConfig

@dataclass(frozen=True)
class SourceFile:
    path: Path
    rel_path: str

def iter_source_files(repo_root: Path, cfg: RepoScanConfig) -> Iterable[SourceFile]:
    repo_root = repo_root.resolve()
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in cfg.exclude_dirs for part in p.parts):
            continue
        if p.suffix.lower() in cfg.include_ext:
            rel = str(p.resolve().relative_to(repo_root))
            yield SourceFile(path=p, rel_path=rel)
