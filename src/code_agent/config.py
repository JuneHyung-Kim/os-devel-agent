from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RepoScanConfig:
    include_ext: tuple[str, ...] = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inl")
    exclude_dirs: tuple[str, ...] = (".git", "build", "cmake-build", "out", "dist", ".cache", ".idea", ".vscode")

@dataclass(frozen=True)
class ChunkConfig:
    # tree-sitter 실패/비정상 파일 대비 백업 라인 청킹
    fallback_max_lines: int = 240
    fallback_overlap_lines: int = 40
    max_chars_per_chunk: int = 40_000  # 너무 큰 함수/헤더 방지

@dataclass(frozen=True)
class Paths:
    data_dir: Path
    chunks_jsonl: Path
    meta_json: Path
    bm25_pkl: Path
    symbol_pkl: Path
    repomap_txt: Path

    @staticmethod
    def from_data_dir(data_dir: Path) -> "Paths":
        data_dir = data_dir.resolve()
        return Paths(
            data_dir=data_dir,
            chunks_jsonl=data_dir / "chunks.jsonl",
            meta_json=data_dir / "meta.json",
            bm25_pkl=data_dir / "bm25.pkl",
            symbol_pkl=data_dir / "symbol.pkl",
            repomap_txt=data_dir / "repo_map.txt",
        )