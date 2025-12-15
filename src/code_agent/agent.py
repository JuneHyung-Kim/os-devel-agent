from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from .retrievers.base import Retriever, RetrievedChunk
from .context_packer import pack_context, PackConfig

EXPLAIN_TEMPLATE = """당신의 목표는 문서 요약이 아니라, 코드의 '존재 이유'를 역할·책임·경계 중심으로 설명하는 것입니다.
반드시 제공된 코드 근거(파일/라인)를 인용하며, 근거가 부족하면 '추정'으로 표시하고 추가로 찾아야 할 근거를 제안하십시오.

출력 형식:
1) 역할(Role): ...
2) 책임(Responsibility): (3~6개) - 각 항목에 근거 라인
3) 경계(Boundary): 외부에 노출/내부 구현/의존성 방향
4) 실행 흐름 위치(Flow): 어디서 호출/등록되는지 (모르면 추정)
5) 왜 존재하는가(Rationale): 분리 이유(성능/격리/호환/확장/안정성 등) + 근거
6) 추가로 확인할 근거(있다면): 심볼/파일/키워드

아래는 컨텍스트입니다:
"""

@dataclass(frozen=True)
class AgentResult:
    query: str
    retrieved: Sequence[RetrievedChunk]
    context: str
    # LLM 연동 시 final_text를 채우면 됨
    final_text: str | None

class ExplainAgent:
    """
    1) Retriever로 근거 수집
    2) Repo map + 근거를 pack
    3) (선택) LLM에 전달하여 역할/책임/경계 설명 생성
    """
    def __init__(self, retriever: Retriever, repo_map_text: str = ""):
        self.retriever = retriever
        self.repo_map_text = repo_map_text

    def run(self, query: str, top_k: int = 10, pack_cfg: PackConfig | None = None) -> AgentResult:
        pack_cfg = pack_cfg or PackConfig()
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        ctx = pack_context(self.repo_map_text, retrieved, pack_cfg)
        # LLM은 사용자가 선택적으로 붙이도록 인터페이스만 남김
        return AgentResult(query=query, retrieved=retrieved, context=ctx, final_text=None)

    def build_llm_prompt(self, ctx: str) -> str:
        return EXPLAIN_TEMPLATE + "\n\n" + ctx
