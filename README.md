# code_agent

- tree-sitter 기반 심볼 단위 청킹
- Repo map 생성
- Retriever 인터페이스 분리(교체 가능)
- 기본: Symbol + BM25 하이브리드
- 인덱스: JSONL + pickle + 메타(json)

## Build
python -m code_agent.cli build --repo /path/to/repo --data ./data_repo

## Ask
python -m code_agent.cli ask --data ./data_repo --query "..." --topk 10
