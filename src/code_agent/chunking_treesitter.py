from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Iterable, Optional
import json

from tree_sitter_languages import get_parser
from .config import ChunkConfig
from .util import safe_read_text

# tree-sitter C/C++에서 의미 있는 노드 타입(최소)
C_LIKE_SYMBOL_NODES = {
    "function_definition",
    "declaration",            # 전역 선언(헤더에서 중요)
    "struct_specifier",
    "enum_specifier",
    "class_specifier",
    "namespace_definition",
}

@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    rel_path: str
    abs_path: str
    language: str            # "c" or "cpp"
    symbol_kind: str         # node.type
    symbol_name: str         # best effort
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    text: str

def _guess_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".c", ".h"):
        return "c"
    return "cpp"

def _node_text(src_bytes: bytes, start_b: int, end_b: int) -> str:
    try:
        return src_bytes[start_b:end_b].decode("utf-8", errors="replace")
    except Exception:
        return ""

def _extract_symbol_name(node, src_bytes: bytes) -> str:
    """
    tree-sitter에서 함수/클래스 이름 추출을 'best effort'로 수행.
    정교한 추출은 다음 단계(쿼리/패턴)로 확장 가능.
    """
    # 함수 정의: declarator 내부 identifier를 찾는 방식
    # C++ 클래스/네임스페이스: name 필드가 identifier일 때가 많음
    # 여기서는 간단히 node 하위에서 첫 identifier를 채택
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type in ("identifier", "type_identifier", "namespace_identifier"):
            return _node_text(src_bytes, cur.start_byte, cur.end_byte).strip()
        for ch in reversed(cur.children):
            stack.append(ch)
    return ""

def _iter_symbol_nodes(root_node) -> Iterable:
    # 전체 트리를 순회하며 관심 노드만 수집
    stack = [root_node]
    while stack:
        n = stack.pop()
        if n.type in C_LIKE_SYMBOL_NODES:
            yield n
        for ch in reversed(n.children):
            stack.append(ch)

def _fallback_line_chunks(rel_path: str, abs_path: str, text: str, lang: str, cfg: ChunkConfig) -> Iterator[CodeChunk]:
    lines = text.splitlines()
    n = len(lines)
    i = 0
    idx = 0
    while i < n:
        start = i
        end = min(n, i + cfg.fallback_max_lines)
        block = "\n".join(lines[start:end])
        if len(block) > cfg.max_chars_per_chunk:
            block = block[:cfg.max_chars_per_chunk]
        chunk_id = f"{rel_path}:{start+1}-{end}:fallback:{idx}"
        yield CodeChunk(
            chunk_id=chunk_id,
            rel_path=rel_path,
            abs_path=abs_path,
            language=lang,
            symbol_kind="fallback_block",
            symbol_name="",
            start_line=start + 1,
            end_line=end,
            start_byte=0,
            end_byte=0,
            text=block,
        )
        idx += 1
        i = end - cfg.fallback_overlap_lines
        if i <= start:
            i = end

def chunk_file_with_treesitter(repo_root: Path, rel_path: str, abs_path: Path, cfg: ChunkConfig) -> list[CodeChunk]:
    lang = _guess_language(abs_path)
    text = safe_read_text(abs_path)
    if text is None:
        return []

    src_bytes = text.encode("utf-8", errors="replace")

    # parser 선택
    try:
        parser = get_parser("c" if lang == "c" else "cpp")
        tree = parser.parse(src_bytes)
        root = tree.root_node
    except Exception:
        # 파서 실패 시 백업
        return list(_fallback_line_chunks(rel_path, str(abs_path), text, lang, cfg))

    chunks: list[CodeChunk] = []
    idx = 0
    for node in _iter_symbol_nodes(root):
        start_b, end_b = node.start_byte, node.end_byte
        if end_b <= start_b:
            continue
        snippet = _node_text(src_bytes, start_b, end_b).strip()
        if not snippet:
            continue
        if len(snippet) > cfg.max_chars_per_chunk:
            snippet = snippet[:cfg.max_chars_per_chunk]

        (srow, _), (erow, _) = node.start_point, node.end_point
        start_line = int(srow) + 1
        end_line = int(erow) + 1

        sym_name = _extract_symbol_name(node, src_bytes)
        chunk_id = f"{rel_path}:{start_line}-{end_line}:{node.type}:{idx}"
        chunks.append(CodeChunk(
            chunk_id=chunk_id,
            rel_path=rel_path,
            abs_path=str(abs_path),
            language=lang,
            symbol_kind=node.type,
            symbol_name=sym_name,
            start_line=start_line,
            end_line=end_line,
            start_byte=start_b,
            end_byte=end_b,
            text=snippet,
        ))
        idx += 1

    # 심볼이 거의 없으면(예: 거대 매크로 헤더) 백업 청킹 추가
    if len(chunks) < 3:
        chunks = list(_fallback_line_chunks(rel_path, str(abs_path), text, lang, cfg))

    return chunks

def write_chunks_jsonl(chunks: Iterable[CodeChunk], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n

def read_chunks_jsonl(path: Path) -> list[CodeChunk]:
    out: list[CodeChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(CodeChunk(**d))
    return out
