"""工具执行器：模型想调工具 和 工具真的被执行 之间的那道闸门。

为什么把执行单独抽出来（而不是写在 Agent 循环里）？
  因为本项目有**两种**调用方：
    · 确定性工作流（8 类高频意图）—— 不经过模型，直接按步骤调工具
    · 自主 Agent（长尾）—— 模型决定调什么
  这两条路径**必须共用同一套**护栏、幂等、Trace、审计。
  如果各写一份，早晚会漂移成"工作流有护栏、Agent 没有"这种事故。

关键设计：**HITL 不在执行器里阻塞等待**。
  interrput 必须发生在**独立节点**里，否则恢复时会把"模型调用 + 前置工具"整段重跑，
  既重复计费又可能产生不同的工具调用。所以执行器的职责是：
    · 检查 → 能执行就执行
    · 需要审批 → 返回 requires_approval，由调用方路由到专门的审批节点
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from .config import Settings, settings as global_settings
from .guardrail import Decision, Guardrail
from .schemas import Step, ToolResult
from .tools.base import Tool, ToolContext, get_tool


@dataclass
class Prepared:
    """一次工具调用的"预检结果"。"""

    tool: Tool | None
    decision: Decision
    blocked_result: ToolResult | None = None

    @property
    def ready(self) -> bool:
        return self.tool is not None and self.decision.allow and self.blocked_result is None


class ToolExecutor:
    def __init__(
        self,
        settings: Settings | None = None,
        guardrail: Guardrail | None = None,
        tracer: Any = None,
        repo: Any = None,
        profile: Any = None,
        session_memory: Any = None,
    ) -> None:
        self.s = settings or global_settings
        self.guardrail = guardrail or Guardrail(self.s)
        self.tracer = tracer
        self.repo = repo
        self.profile = profile
        self.session_memory = session_memory

    # ------------------------------------------------------------- 预检
    def prepare(self, state: Any, name: str, arguments: dict[str, Any]) -> Prepared:
        tool = get_tool(name)
        if tool is None:
            reason = f"没有名为 {name} 的工具（模型可能记错了工具名）"
            return Prepared(
                None, Decision(False, reason, policy="unknown_tool"),
                ToolResult(call_id="", name=name, ok=False, blocked=True, block_reason=reason),
            )

        ctx = self._run_context(state)
        decision = self.guardrail.check(tool, arguments or {}, state, ctx)
        if not decision.allow:
            return Prepared(
                tool, decision,
                ToolResult(call_id="", name=tool.name, ok=False, blocked=True,
                           block_reason=decision.reason),
            )
        return Prepared(tool, decision)

    # ------------------------------------------------------------- 执行
    def call(
        self, state: Any, name: str, arguments: dict[str, Any], approved: bool = False,
        call_id: str = "",
    ) -> ToolResult:
        prepared = self.prepare(state, name, arguments)

        if prepared.blocked_result is not None:
            prepared.blocked_result.call_id = call_id
            self._record_blocked(state, name, arguments, prepared.decision)
            return prepared.blocked_result

        tool = prepared.tool
        assert tool is not None

        # 需要人工审批，且调用方还没拿到审批结果 → 交回控制权，由审批节点处理
        if prepared.decision.needs_confirm and not approved:
            result = ToolResult(
                call_id=call_id, name=tool.name, ok=False, requires_approval=True,
                block_reason=prepared.decision.reason,
            )
            self._record_step(
                state, "approval",
                {"tool": tool.name, "arguments": arguments, "status": "pending",
                 "reason": prepared.decision.reason, "risk_level": prepared.decision.risk_level},
            )
            return result

        return self._execute(state, tool, arguments, call_id, approved=approved,
                             decision=prepared.decision)

    def _execute(
        self, state: Any, tool: Tool, arguments: dict[str, Any], call_id: str,
        approved: bool, decision: Decision,
    ) -> ToolResult:
        started = time.time()
        ctx = ToolContext(
            run_id=getattr(state, "run_id", ""),
            thread_id=getattr(state, "thread_id", ""),
            session_id=getattr(state, "session_id", ""),
            intent=getattr(state, "intent", ""),
            repo=self.repo, profile=self.profile, session_memory=self.session_memory,
        )
        try:
            output = self._call_with_timeout(tool, arguments, ctx)
            result = ToolResult(
                call_id=call_id, name=tool.name, ok=True,
                output=self.guardrail.sanitize(str(output)),
                confirmed=True if approved else None,
                duration_ms=int((time.time() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 —— 工具报错要回传给模型，而不是崩掉图
            result = ToolResult(
                call_id=call_id, name=tool.name, ok=False,
                error=f"{type(exc).__name__}: {exc}",
                confirmed=True if approved else None,
                duration_ms=int((time.time() - started) * 1000),
            )

        self._count(state, tool, arguments)
        self._record_step(
            state, "tool_call",
            {
                "tool": tool.name, "arguments": arguments, "ok": result.ok,
                "output": (result.output or "")[:800], "error": result.error,
                "approved": approved or None, "risk_level": tool.risk_level,
                "policy": decision.policy,
            },
            result.duration_ms,
        )
        if self.tracer is not None:
            self.tracer.tool_call(
                tool.name, arguments, result.output or result.error, result.ok,
                result.duration_ms, blocked=False,
                approved=True if approved else None,
            )
        return result

    # ------------------------------------------------------------- 便捷方法
    def call_many(self, state: Any, calls: list[tuple[str, dict]]) -> list[ToolResult]:
        return [self.call(state, name, args) for name, args in calls]

    def safe_call(self, state: Any, name: str, arguments: dict, approved: bool = False) -> str:
        """返回文本结果，方便工作流里串流程。"""
        return self.call(state, name, arguments, approved=approved).to_llm_content()

    # ------------------------------------------------------------- 内部
    @staticmethod
    def _call_with_timeout(tool: Tool, arguments: dict, ctx: ToolContext) -> str:
        """带超时执行。

        已知局限：Python 线程无法被强杀，超时后线程仍在后台跑。
        生产环境应把工具执行放到独立进程/容器（本项目的数据分析工具已走沙箱进程）。
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(tool.call, arguments, ctx)
            try:
                return future.result(timeout=tool.timeout)
            except FutureTimeout as exc:
                raise TimeoutError(f"工具执行超过 {tool.timeout:.0f} 秒，已放弃") from exc

    @staticmethod
    def _run_context(state: Any):
        from .guardrail import RunContext

        return RunContext(
            tool_calls=getattr(state, "tool_calls", 0),
            blocked=getattr(state, "blocked", 0),
            call_counter=getattr(state, "call_counter", {}) or {},
        )

    @staticmethod
    def _count(state: Any, tool: Tool, arguments: dict) -> None:
        state.tool_calls = getattr(state, "tool_calls", 0) + 1
        signature = f"{tool.name}:{sorted(arguments.items())}"
        counter = getattr(state, "call_counter", None)
        if isinstance(counter, dict):
            counter[signature] = counter.get(signature, 0) + 1

    def _record_blocked(self, state: Any, name: str, arguments: dict, decision: Decision) -> None:
        state.blocked = getattr(state, "blocked", 0) + 1
        self._record_step(
            state, "guardrail",
            {"tool": name, "arguments": arguments, "blocked": True,
             "reason": decision.reason, "policy": decision.policy, "level": "WARNING"},
        )
        if self.tracer is not None:
            self.tracer.tool_call(name, arguments, "", False, 0, blocked=True,
                                  reason=decision.reason)

    @staticmethod
    def _record_step(state: Any, kind: str, detail: dict, duration_ms: int = 0) -> None:
        steps = getattr(state, "steps", None)
        if steps is None:
            return
        steps.append(Step(index=len(steps) + 1, type=kind, duration_ms=duration_ms,
                          detail=_jsonable(detail)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)[:400]


def summarize_arguments(arguments: dict) -> str:
    return json.dumps(arguments, ensure_ascii=False)[:200]
