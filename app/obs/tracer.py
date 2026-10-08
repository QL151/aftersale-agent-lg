"""全链路 Trace：本地落盘 + 可选镜像到 Langfuse。

设计取舍（很重要）：
  **本地 JSONL 永远是主存储**，Langfuse 是"镜像"。
  为什么？因为如果 Langfuse 是唯一存储，那么：
    · 平台挂了 → 你的排障能力同时挂掉
    · 新人 clone 项目 → 没配 Key 就看不到任何 Trace，项目没法演示
  所以本地 trace 是"底线"，Langfuse 是"放大镜"（UI、跨服务串联、标注、打分）。

Trace 里记什么（对应 Bad Case 定位要用的字段）：
  · 路由决策：意图、置信度、走工作流还是自主 Agent
  · 每个节点：进入/退出、耗时、状态变化
  · 每次工具调用：参数、结果、是否被拦、是否走人工审批
  · 每次模型调用：模型名、token、成本、是否降级
  · 结论：结束原因、是否转人工、是否全自动闭环
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import settings
from ..runtime import get_langfuse


@dataclass
class TraceEvent:
    ts: float
    kind: str                  # run / node / tool / llm / approval / cost / final / error
    name: str
    detail: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    level: str = "DEFAULT"     # DEBUG / DEFAULT / WARNING / ERROR

    def to_dict(self) -> dict:
        return {
            "ts": round(self.ts, 3),
            "kind": self.kind,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "level": self.level,
            **self.detail,
        }


class Tracer:
    """一次运行的追踪器。支持 with 语法，异常也会被记录。"""

    def __init__(self, run_id: str, task: str, session_id: str = "",
                 trace_dir: Path | None = None, mirror: bool = True,
                 prior: dict | None = None) -> None:
        self.run_id = run_id
        self.task = task
        self.session_id = session_id
        self.dir = trace_dir or settings.trace_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events: list[TraceEvent] = []
        # 恢复（resume）时传入上一次的 trace：事件要**追加**而不是覆盖。
        # 否则"中断前发生了什么"会整段丢失 —— 而那段恰恰是审批人最需要看的上下文。
        self.prior_events: list[dict] = list((prior or {}).get("events", []))
        self.metadata: dict[str, Any] = dict((prior or {}).get("metadata", {}))
        self.scores: list[dict[str, Any]] = list((prior or {}).get("scores", []))
        self.started = (prior or {}).get("_started_ts") or time.time()
        self.backend = "local"
        self._lf = None
        self._lf_trace = None

        if mirror:
            self._init_langfuse()

    # ------------------------------------------------------------ Langfuse
    def _init_langfuse(self) -> None:
        client = get_langfuse()
        if client is None:
            return
        self._lf = client
        self.backend = "langfuse+local"
        # Langfuse SDK 各版本方法名有差异，全部防御性调用 ——
        # 观测层绝不能因为 API 版本不同把业务搞挂。
        for method in ("trace", "start_trace"):
            fn = getattr(client, method, None)
            if fn is None:
                continue
            try:
                self._lf_trace = fn(
                    id=self.run_id, name="aftersale-agent", input={"task": self.task},
                    metadata={"session_id": self.session_id},
                )
                break
            except Exception:  # noqa: BLE001
                continue

    def _lf_span(self, name: str, **kwargs: Any) -> Any:
        if self._lf is None:
            return None
        for owner in (self._lf_trace, self._lf):
            if owner is None:
                continue
            fn = getattr(owner, "span", None) or getattr(owner, "start_span", None)
            if fn is None:
                continue
            try:
                return fn(name=name, **kwargs)
            except Exception:  # noqa: BLE001
                continue
        return None

    # ------------------------------------------------------------ 记录
    def event(self, kind: str, name: str, detail: dict | None = None,
              duration_ms: int = 0, level: str = "DEFAULT") -> None:
        item = TraceEvent(time.time(), kind, name, detail or {}, duration_ms, level)
        self.events.append(item)
        if kind in {"tool", "llm", "approval"}:
            self._lf_span(name, input=item.detail.get("arguments") or item.detail.get("input"),
                          output=item.detail.get("output"), metadata=item.detail)

    def node(self, name: str, duration_ms: int, detail: dict | None = None) -> None:
        self.event("node", name, detail, duration_ms)

    def tool_call(self, name: str, arguments: dict, result: str, ok: bool,
                  duration_ms: int, blocked: bool = False, reason: str = "",
                  approved: bool | None = None) -> None:
        self.event(
            "tool", name,
            {
                "arguments": arguments, "output": (result or "")[:2000], "ok": ok,
                "blocked": blocked, "block_reason": reason, "approved": approved,
            },
            duration_ms,
            level="WARNING" if blocked else ("ERROR" if not ok else "DEFAULT"),
        )

    def llm_call(self, model: str, prompt_tokens: int, completion_tokens: int,
                 duration_ms: int, text: str = "", tool_calls: int = 0,
                 cost: float = 0.0, downgraded: bool = False) -> None:
        self.event(
            "llm", f"llm:{model}",
            {
                "model": model, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "text": (text or "")[:500],
                "tool_calls": tool_calls, "cost_yuan": round(cost, 6),
                "downgraded": downgraded,
            },
            duration_ms,
        )

    def approval(self, approval_id: str, action: str, decision: str,
                 amount: float = 0.0, reason: str = "") -> None:
        self.event(
            "approval", action,
            {"approval_id": approval_id, "decision": decision, "amount": amount,
             "reason": reason},
            level="WARNING" if decision != "approved" else "DEFAULT",
        )

    def cost(self, total_tokens: int, cost_yuan: float, ratio: float, downgraded: bool) -> None:
        self.event(
            "cost", "budget",
            {"total_tokens": total_tokens, "cost_yuan": round(cost_yuan, 6),
             "ratio": round(ratio, 3), "downgraded": downgraded},
            level="ERROR" if ratio >= 1.0 else ("WARNING" if ratio >= settings.cost_warn_ratio else "DEFAULT"),
        )

    def score(self, name: str, value: float, comment: str = "") -> None:
        self.scores.append({"name": name, "value": value, "comment": comment})
        if self._lf is None:
            return
        for method in ("score", "create_score"):
            fn = getattr(self._lf, method, None)
            if fn is None:
                continue
            try:
                fn(name=name, value=value, comment=comment, trace_id=self.run_id)
                return
            except Exception:  # noqa: BLE001
                continue

    # ------------------------------------------------------------ 收尾
    def finish(self, result: Any) -> Path:
        """把整次运行写成一份 JSON（可回放）+ 一行摘要（可列表/统计）。"""
        payload = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "task": self.task,
            "backend": self.backend,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
            "total_ms": int((time.time() - self.started) * 1000),
            "result": result.to_dict() if hasattr(result, "to_dict") else result,
            "metadata": self.metadata,
            "scores": self.scores,
            "events": [*self.prior_events, *[e.to_dict() for e in self.events]],
        }
        path = self.dir / f"{self.run_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "task": self.task,
            "intent": _get(result, "intent"),
            "route": _get(result, "route"),
            "workflow": _get(result, "workflow"),
            "stopped_reason": _get(result, "stopped_reason"),
            "answer": (_get(result, "answer") or "")[:160],
            "tool_calls": _get(result, "tool_calls"),
            "blocked_calls": _get(result, "blocked_calls"),
            "total_tokens": _get(result, "total_tokens"),
            "cost_yuan": round(float(_get(result, "cost_yuan") or 0.0), 6),
            "total_ms": _get(result, "total_ms"),
            "auto_closed": bool(_get(result, "auto_closed")),
            "handover": bool(_get(result, "handover")),
            "downgraded": bool(_get(result, "downgraded")),
            "approvals": _get(result, "approvals"),
            "engine": _get(result, "engine"),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (self.dir / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

        if self._lf is not None:
            for method in ("flush", "update_current_trace"):
                fn = getattr(self._lf, method, None)
                if fn is None:
                    continue
                try:
                    fn(output={"answer": _get(result, "answer")})
                except Exception:  # noqa: BLE001
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        pass
        return path


def _get(obj: Any, name: str) -> Any:
    return getattr(obj, name, None) if not isinstance(obj, dict) else obj.get(name)


# ==================================================================== 读取工具


def load_traces(trace_dir: Path | None = None, limit: int = 100) -> list[dict]:
    directory = trace_dir or settings.trace_dir
    index = directory / "index.jsonl"
    if not index.exists():
        return []
    rows: list[dict] = []
    for line in index.read_text(encoding="utf-8").splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # 某一行坏了不影响其它记录
    return list(reversed(rows))


def load_trace(run_id: str, trace_dir: Path | None = None) -> dict | None:
    directory = trace_dir or settings.trace_dir
    path = directory / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def new_run_id() -> str:
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def current_env() -> dict:
    return {
        "pid": os.getpid(),
        "engine": settings.graph_engine,
    }
