"""
Agent Trace Callback Handler

Provides real-time console streaming and structured file logging for
all LLM calls, tool invocations, and node transitions — without LangSmith.

Output:
  Console : colored, truncated for readability
  File    : ./logs/traces/trace_YYYYMMDD_HHMMSS.jsonl (full content, one JSON per line)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

# ── ANSI colors ───────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_BLUE   = "\033[34m"
_MAGENTA= "\033[35m"
_RED    = "\033[31m"
_GRAY   = "\033[90m"

_CONSOLE_MAX_LEN = 800   # truncate long text in console


def _truncate(text: str, max_len: int = _CONSOLE_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n{_GRAY}  … ({len(text) - max_len} chars truncated){_RESET}"


def _format_messages(messages: List[BaseMessage]) -> str:
    """Format a list of LangChain messages for display."""
    parts = []
    for msg in messages:
        role = type(msg).__name__.replace("Message", "").upper()
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


class AgentTraceCallback(BaseCallbackHandler):
    """
    LangChain callback handler for full agent observability.

    Args:
        trace_dir:        Directory for trace files.
        stream_to_console: Print events to stdout in real-time.
        color:            Enable ANSI colors in console output.
    """

    def __init__(
        self,
        trace_dir: str = "./logs/traces",
        stream_to_console: bool = True,
        color: bool = True,
    ):
        super().__init__()
        self.stream_to_console = stream_to_console
        self.color = color
        self._start_times: Dict[str, float] = {}  # run_id → start time

        # ── Trace file setup ──────────────────────────────────────────────────
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_path = Path(trace_dir) / f"trace_{session_ts}.jsonl"
        self._trace_file = open(self.trace_path, "w", encoding="utf-8")

        self._emit("session_start", {"trace_file": str(self.trace_path)})
        if stream_to_console:
            self._console(
                f"\n{'━'*60}\n"
                f"  AGENT TRACE  |  {session_ts}\n"
                f"  Log → {self.trace_path}\n"
                f"{'━'*60}",
                _CYAN,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _c(self, code: str) -> str:
        return code if self.color else ""

    def _console(self, text: str, color: str = "") -> None:
        if self.stream_to_console:
            print(f"{self._c(color)}{text}{self._c(_RESET)}", flush=True)

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        record = {
            "ts": datetime.now().isoformat(),
            "event": event_type,
            **data,
        }
        self._trace_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._trace_file.flush()

    def _elapsed(self, run_id: UUID) -> str:
        key = str(run_id)
        if key in self._start_times:
            secs = time.perf_counter() - self._start_times.pop(key)
            return f"{secs:.2f}s"
        return "?"

    # ── LangGraph node transitions (on_chain_*) ───────────────────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name") or serialized.get("id", ["?"])[-1]
        self._start_times[str(run_id)] = time.perf_counter()
        self._emit("node_start", {"node": name, "run_id": str(run_id)})
        self._console(f"\n┌── NODE: {name} {'─'*max(0,50-len(name))}", _CYAN + _BOLD)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed = self._elapsed(run_id)
        self._emit("node_end", {"run_id": str(run_id), "elapsed": elapsed})
        self._console(f"└── done ({elapsed})", _GRAY)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed = self._elapsed(run_id)
        self._emit("node_error", {"error": str(error), "run_id": str(run_id), "elapsed": elapsed})
        self._console(f"└── ERROR: {error}", _RED)

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model_name = serialized.get("name") or serialized.get("kwargs", {}).get("model", "LLM")
        self._start_times[str(run_id)] = time.perf_counter()

        # Flatten all message turns for this call
        flat = [m for turn in messages for m in turn]
        prompt_text = _format_messages(flat)

        self._emit("llm_start", {
            "model": model_name,
            "run_id": str(run_id),
            "messages": [{"role": type(m).__name__, "content": m.content} for m in flat],
        })
        self._console(
            f"  ▶ LLM call [{model_name}]\n"
            f"{_GRAY}  {'─'*56}{_RESET}\n"
            f"{_YELLOW}{_truncate(prompt_text)}{_RESET}",
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed = self._elapsed(run_id)
        generations = response.generations
        texts = []
        tool_calls_all = []

        for gen_list in generations:
            for gen in gen_list:
                text = getattr(gen, "text", "") or ""
                msg  = getattr(gen, "message", None)
                if msg:
                    # Capture tool_calls if present (function calling)
                    tc = getattr(msg, "tool_calls", []) or []
                    if tc:
                        tool_calls_all.extend(tc)
                    if not text and hasattr(msg, "content"):
                        text = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                if text:
                    texts.append(text)

        combined = "\n".join(texts)
        token_usage = response.llm_output.get("token_usage") if response.llm_output else None

        self._emit("llm_end", {
            "run_id": str(run_id),
            "elapsed": elapsed,
            "response": combined,
            "tool_calls": tool_calls_all,
            "token_usage": token_usage,
        })

        # Console: response text
        if combined:
            self._console(
                f"  ◀ LLM response ({elapsed})\n"
                f"{_GRAY}  {'─'*56}{_RESET}\n"
                f"{_GREEN}{_truncate(combined)}{_RESET}",
            )

        # Console: tool calls (structured output)
        if tool_calls_all:
            for tc in tool_calls_all:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                self._console(
                    f"  ◀ Tool call [{name}]  ({elapsed})\n"
                    f"{_GRAY}  {json.dumps(args, ensure_ascii=False, indent=4)}{_RESET}",
                )

        # Token usage
        if token_usage:
            self._console(f"  {_GRAY}tokens: {token_usage}{_RESET}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed = self._elapsed(run_id)
        self._emit("llm_error", {"error": str(error), "run_id": str(run_id), "elapsed": elapsed})
        self._console(f"  ✗ LLM error ({elapsed}): {error}", _RED)

    # ── Tool calls ────────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "tool")
        self._start_times[str(run_id)] = time.perf_counter()
        self._emit("tool_start", {"tool": name, "input": input_str, "run_id": str(run_id)})
        self._console(
            f"\n  ▷ Tool [{name}]\n"
            f"{_BLUE}{_truncate(input_str, 400)}{_RESET}",
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed = self._elapsed(run_id)
        output_str = str(output)
        self._emit("tool_end", {"output": output_str, "elapsed": elapsed, "run_id": str(run_id)})
        self._console(
            f"  ◁ Tool result ({elapsed})\n"
            f"{_MAGENTA}{_truncate(output_str, 600)}{_RESET}",
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed = self._elapsed(run_id)
        self._emit("tool_error", {"error": str(error), "elapsed": elapsed, "run_id": str(run_id)})
        self._console(f"  ✗ Tool error ({elapsed}): {error}", _RED)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the trace file."""
        self._emit("session_end", {})
        self._trace_file.close()
        if self.stream_to_console:
            self._console(
                f"\n{'━'*60}\n"
                f"  TRACE COMPLETE  →  {self.trace_path}\n"
                f"{'━'*60}\n",
                _CYAN,
            )
