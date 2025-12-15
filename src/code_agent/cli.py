from __future__ import annotations
import argparse
from pathlib import Path
from tqdm import tqdm

from .config import RepoScanConfig, ChunkConfig, Paths
from .repo_scan import iter_source_files
from .chunking_treesitter import chunk_file_with_treesitter, write_chunks_jsonl, read_chunks_jsonl, CodeChunk
from .util import sha1_file
from .store import IndexMeta, build_symbol_index
from .repo_map import build_repo_map, RepoMapConfig

from .retrievers import SymbolRetriever, BM25Retriever, BM25Store, HybridRetriever
from .retrievers.bm25 import tokenize_code
from .store import SymbolIndex
from .agent import ExplainAgent

def _load_meta(paths: Paths) -> IndexMeta | None:
    if not paths.meta_json.exists():
        return None
    return IndexMeta.load(paths.meta_json)

def _build_index(repo_root: Path, data_dir: Path) -> None:
    paths = Paths.from_data_dir(data_dir)
    scan_cfg = RepoScanConfig()
    chunk_cfg = ChunkConfig()

    repo_root = repo_root.resolve()
    old_meta = _load_meta(paths)
    old_hashes = old_meta.file_hashes if old_meta else {}

    files = list(iter_source_files(repo_root, scan_cfg))
    new_hashes: dict[str, str] = {}
    changed: list[tuple[str, Path]] = []

    for sf in tqdm(files, desc="hash"):
        h = sha1_file(sf.path)
        new_hashes[sf.rel_path] = h
        if old_hashes.get(sf.rel_path) != h:
            changed.append((sf.rel_path, sf.path))

    # 기존 청크 로드(있으면)
    existing_chunks: list[CodeChunk] = []
    if paths.chunks_jsonl.exists():
        existing_chunks = read_chunks_jsonl(paths.chunks_jsonl)

    # 변경 파일 청크는 재생성, 나머지는 유지
    changed_set = set(rp for rp, _ in changed)
    kept_chunks = [c for c in existing_chunks if c.rel_path not in changed_set]

    new_chunks: list[CodeChunk] = []
    for rel_path, abs_path in tqdm(changed, desc="parse"):
        new_chunks.extend(chunk_file_with_treesitter(repo_root, rel_path, abs_path, chunk_cfg))

    all_chunks = kept_chunks + new_chunks

    # 저장
    n = write_chunks_jsonl(all_chunks, paths.chunks_jsonl)

    meta = IndexMeta(repo_root=str(repo_root), file_hashes=new_hashes, chunk_count=n)
    meta.save(paths.meta_json)

    # symbol index 저장
    sym = build_symbol_index(all_chunks)
    sym.save(paths.symbol_pkl)

    # bm25 저장
    tokenized = [tokenize_code(c.text) for c in tqdm(all_chunks, desc="bm25 tokenize")]
    BM25Store(tokenized_corpus=tokenized, chunks=all_chunks).save(paths.bm25_pkl)

    # repo map 저장
    repomap = build_repo_map(all_chunks, RepoMapConfig())
    paths.repomap_txt.parent.mkdir(parents=True, exist_ok=True)
    paths.repomap_txt.write_text(repomap, encoding="utf-8")

    print(f"OK: chunks={n}, changed_files={len(changed)}, data_dir={paths.data_dir}")

def _ask(data_dir: Path, query: str, top_k: int) -> None:
    paths = Paths.from_data_dir(data_dir)
    if not paths.bm25_pkl.exists() or not paths.symbol_pkl.exists() or not paths.chunks_jsonl.exists():
        raise SystemExit("Index not found. Run build first.")

    chunks = read_chunks_jsonl(paths.chunks_jsonl)
    sym = SymbolIndex.load(paths.symbol_pkl)
    bm25_store = BM25Store.load(paths.bm25_pkl)

    # Retriever 구성(교체 지점)
    sym_r = SymbolRetriever(chunks=chunks, sym_index=sym)
    bm25_r = BM25Retriever(store=bm25_store)
    retriever = HybridRetriever(a=sym_r, b=bm25_r, weight_a=2.0, weight_b=1.0)

    repo_map_text = paths.repomap_txt.read_text(encoding="utf-8") if paths.repomap_txt.exists() else ""

    agent = ExplainAgent(retriever=retriever, repo_map_text=repo_map_text)
    res = agent.run(query=query, top_k=top_k)

    print("\n[Retrieved]")
    for r in res.retrieved:
        print(f"- {r.rel_path}:{r.start_line}-{r.end_line} score={r.score:.3f} ({r.reason})")

    print("\n[LLM Prompt Preview (context packed)]\n")
    prompt = agent.build_llm_prompt(res.context)
    print(prompt[:8000])  # 너무 길면 일부만 출력

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build/Update index (tree-sitter symbol chunks + repo map + bm25 + symbol index)")
    b.add_argument("--repo", required=True, help="Path to repo root")
    b.add_argument("--data", required=True, help="Output data dir")

    a = sub.add_parser("ask", help="Retrieve context for a query")
    a.add_argument("--data", required=True, help="Data dir")
    a.add_argument("--query", required=True, help="Question")
    a.add_argument("--topk", type=int, default=10)

    args = ap.parse_args()

    if args.cmd == "build":
        _build_index(repo_root=Path(args.repo), data_dir=Path(args.data))
    elif args.cmd == "ask":
        _ask(data_dir=Path(args.data), query=args.query, top_k=args.topk)

if __name__ == "__main__":
    main()
