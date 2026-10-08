"""Bad Case 定位：把"翻 2 小时日志"变成"一条命令 15 分钟看完"。

为什么单独做这个模块？
  因为 Agent 上线后 90% 的时间花在"这条为什么答错了"。
  而 Agent 的失败**不是异常栈**，是"第 3 步模型理解偏了、第 5 步工具被拦了、
  第 7 步成本熔断把模型换成小的所以答糊了"——这些都躺在 Trace 里，
  但没人愿意手工翻 JSON。

  所以这里做两件事：
    ① list_bad_cases()  —— 批量筛出"值得看"的 run，并按失败类型归类
    ② explain(run_id)   —— 单条 run 的因果链 + 根因判定 + 修复建议
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import settings
from .tracer import load_trace, load_traces


@dataclass
class Issue:
    """一条"值得人工看一眼"的 run。"""

    run_id: str
    task: str
    category: str            # 归因分类，见 CATEGORY_RULES
    severity: str            # high / medium / low
    evidence: list[str] = field(default_factory=list)
    suggestion: str = ""

    def line(self) -> str:
        return f"[{self.severity.upper():<6}] {self.category:<22} {self.run_id}  {self.task[:44]}"


CATEGORY_RULES: dict[str, tuple[str, str]] = {
    # category: (severity, suggestion)
    "loop_not_converged": ("high", "循环未收敛：检查是否缺少终止条件，或工具返回内容让模型无法判断已完成"),
    "timeout": ("high", "超时：检查是哪个工具慢，或考虑把长任务拆成异步工单"),
    "node_error": ("high", "节点异常：看 events 里第一条 level=ERROR 的记录，通常是工具抛错或数据缺失"),
    "guardrail_blocked": ("medium", "被护栏拦截：确认是误拦（放宽策略）还是模型越权（改提示词/工具描述）"),
    "approval_denied": ("medium", "人工驳回：看审批备注，通常是金额/依据不充分，需要补证据再提交"),
    "cost_downgraded": ("medium", "触发了成本熔断降级：检查会话 token 预算是否过紧，或提示词是否太长"),
    "route_low_confidence": ("medium", "路由置信度低、走了自主 Agent：考虑补一条确定性工作流，或补关键词"),
    "handover": ("low", "转人工：若是设计内兜底可忽略；若比例偏高说明工作流覆盖不足"),
    "empty_answer": ("high", "结论为空：检查模型是否被工具返回的大段文本淹没"),
}


def classify(summary: dict, events: list[dict] | None = None) -> list[Issue]:
    """给一条 trace 摘要打标签（可多标签）。"""
    issues: list[Issue] = []
    run_id = summary.get("run_id", "")
    task = summary.get("task", "")
    stopped = summary.get("stopped_reason", "")
    events = events or []

    def add(category: str, evidence: list[str]) -> None:
        severity, suggestion = CATEGORY_RULES[category]
        issues.append(Issue(run_id, task, category, severity, evidence, suggestion))

    if stopped == "max_steps":
        add("loop_not_converged", [f"stopped_reason={stopped}"])
    elif stopped == "timeout":
        add("timeout", [f"stopped_reason={stopped}", f"total_ms={summary.get('total_ms')}"])
    elif stopped == "error":
        errs = [e for e in events if e.get("level") == "ERROR"]
        add("node_error", [e.get("name", "?") + ": " + str(e.get("error", ""))[:120] for e in errs[:3]]
            or [f"stopped_reason={stopped}"])

    if summary.get("blocked_calls"):
        blocked = [e for e in events if e.get("kind") == "tool" and e.get("blocked")]
        add("guardrail_blocked",
            [f"{e.get('name')} 被拦：{str(e.get('block_reason'))[:100]}" for e in blocked[:3]]
            or [f"blocked_calls={summary['blocked_calls']}"])

    denied = [e for e in events if e.get("kind") == "approval" and e.get("decision") != "approved"]
    if denied:
        add("approval_denied",
            [f"{e.get('name')} 审批结果={e.get('decision')} 金额={e.get('amount')}" for e in denied[:3]])

    if summary.get("downgraded"):
        add("cost_downgraded", [f"total_tokens={summary.get('total_tokens')}"])

    route_events = [e for e in events if e.get("kind") == "route"]
    for event in route_events:
        confidence = event.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < settings.router_confidence_threshold:
            add("route_low_confidence",
                [f"intent={event.get('intent')} confidence={confidence:.2f} "
                 f"< 阈值 {settings.router_confidence_threshold}"])

    if summary.get("handover"):
        add("handover", ["handover=true"])

    if not (summary.get("answer") or "").strip():
        add("empty_answer", ["answer 为空"])

    return issues


def list_bad_cases(limit: int = 200, trace_dir: Path | None = None) -> list[Issue]:
    """扫全部 trace，筛出需要看的 run。"""
    directory = trace_dir or settings.trace_dir
    issues: list[Issue] = []
    for summary in load_traces(directory, limit=limit):
        trace = load_trace(summary.get("run_id", ""), directory)
        events = (trace or {}).get("events", [])
        issues.extend(classify(summary, events))
    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: order.get(i.severity, 3))
    return issues


def latest_run_id(trace_dir: Path | None = None) -> str:
    """最近一条「值得看」的 run：优先最近的 Bad Case，没有就退回最近一条 Trace。

    为什么要这个：run_id 是 `run-20261008-160438-6780` 这种长串，让人手抄一遍
    既费事又容易错（Windows 的 PowerShell 还会把 `<run_id>` 当成保留运算符报错）。
    run_id 中间就是时间戳，字典序 == 时间序，所以 max() 拿到的就是最新那条。
    """
    issues = list_bad_cases(trace_dir=trace_dir)
    if issues:
        return max(issue.run_id for issue in issues)
    recent = load_traces(trace_dir, limit=1)
    return recent[0].get("run_id", "") if recent else ""


def bad_case_report(limit: int = 200, trace_dir: Path | None = None) -> str:
    issues = list_bad_cases(limit, trace_dir)
    if not issues:
        return "没有发现 Bad Case。所有 trace 都正常收尾。"

    by_category: dict[str, int] = {}
    for issue in issues:
        by_category[issue.category] = by_category.get(issue.category, 0) + 1

    lines = ["=" * 96, f"Bad Case 扫描报告（共 {len(issues)} 条待看，按严重度排序）", "=" * 96]
    lines.append("分类统计：")
    for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
        severity, suggestion = CATEGORY_RULES.get(category, ("?", ""))
        lines.append(f"  {count:>4} 条  [{severity:<6}] {category:<22} → {suggestion}")
    lines.append("-" * 96)
    lines.append("明细（前 25 条）：")
    for issue in issues[:25]:
        lines.append("  " + issue.line())
        for item in issue.evidence[:2]:
            lines.append(f"          · {item}")
    lines.append("-" * 96)
    lines.append("看某一条的完整因果链：python run_cli.py --explain <run_id>   ← 把 <run_id> 换成上面的编号")
    lines.append("不想挑就直接看最近一条：python run_cli.py --explain")
    return "\n".join(lines)


def explain(run_id: str, trace_dir: Path | None = None) -> str:
    """单条 run 的因果链 + 根因 + 建议 —— 这就是"15 分钟定位"的核心。"""
    trace = load_trace(run_id, trace_dir)
    if not trace:
        return f"没有找到 Trace：{run_id}"

    result = trace.get("result", {})
    events = trace.get("events", [])
    summary = {
        "run_id": run_id,
        "task": trace.get("task", ""),
        "stopped_reason": result.get("stopped_reason", ""),
        "blocked_calls": result.get("blocked_calls", 0),
        "handover": result.get("handover"),
        "downgraded": result.get("downgraded"),
        "total_tokens": (result.get("prompt_tokens", 0) + result.get("completion_tokens", 0)),
        "total_ms": result.get("total_ms"),
        "answer": result.get("answer", ""),
    }
    issues = classify(summary, events)

    lines = [
        "=" * 96,
        f"Run 诊断：{run_id}",
        "=" * 96,
        f"任务      ：{trace.get('task', '')}",
        f"会话      ：{trace.get('session_id', '')}",
        f"意图/路由 ：{result.get('intent', '-')} / {result.get('route', '-')}"
        f"{' / ' + result['workflow'] if result.get('workflow') else ''}",
        f"引擎      ：{result.get('engine', '-')}",
        f"结束原因  ：{result.get('stopped_reason', '-')}",
        f"是否闭环  ：自动闭环={bool(result.get('auto_closed'))}  转人工={bool(result.get('handover'))}"
        f"  人工审批={result.get('approvals', 0)}",
        f"用量      ：tokens={summary['total_tokens']}  成本=¥{result.get('cost_yuan', 0)}"
        f"  模型={result.get('model_used', '-')}{'  [已降级]' if result.get('downgraded') else ''}",
        f"耗时      ：{result.get('total_ms', 0)} ms",
        "",
        f"结论：{(result.get('answer') or '')[:200]}",
        "-" * 96,
    ]

    if issues:
        lines.append("根因判定：")
        for issue in issues:
            lines.append(f"  ● [{issue.severity.upper()}] {issue.category}")
            for item in issue.evidence:
                lines.append(f"      · {item}")
            lines.append(f"      → 建议：{issue.suggestion}")
    else:
        lines.append("根因判定：未发现异常特征（这条 run 表现正常）。")

    lines.append("-" * 96)
    lines.append(f"完整因果链（{len(events)} 个事件）：")
    for index, event in enumerate(events, 1):
        mark = {"ERROR": "!!", "WARNING": "!"}.get(event.get("level", ""), "  ")
        detail = _brief(event)
        lines.append(
            f"  {mark} #{index:<3} {event.get('kind', ''):<9} {event.get('name', ''):<24}"
            f"{event.get('duration_ms', 0):>6}ms  {detail}"
        )
    return "\n".join(lines)


def _brief(event: dict) -> str:
    """把事件压成一行关键信息 —— 因果链要看的是"发生了什么"，不是字段全集。"""
    kind = event.get("kind")
    if kind == "route":
        return (f"intent={event.get('intent')} conf={event.get('confidence')} "
                f"→ {event.get('route')}/{event.get('workflow', '-')}")
    if kind == "tool":
        status = "被拦" if event.get("blocked") else ("成功" if event.get("ok") else "失败")
        extra = f" 原因={event.get('block_reason')}" if event.get("blocked") else ""
        return f"{status} args={_short(event.get('arguments'))}{extra} → {_short(event.get('output'), 70)}"
    if kind == "llm":
        return (f"model={event.get('model')} tok={event.get('prompt_tokens')}+"
                f"{event.get('completion_tokens')} tools={event.get('tool_calls')} "
                f"{'[降级]' if event.get('downgraded') else ''}")
    if kind == "approval":
        return f"审批={event.get('decision')} 金额={event.get('amount')} {event.get('reason', '')}"
    if kind == "cost":
        return f"tokens={event.get('total_tokens')} 占比={event.get('ratio')}"
    if kind == "node":
        return _short(event.get("detail") or event.get("answer"), 60)
    return _short(event.get("error") or event.get("answer") or event.get("text"), 70)


def _short(value: Any, limit: int = 60) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def top_bad_case_commands() -> str:
    return (
        "python run_cli.py --bad-cases          # 批量扫描\n"
        "python run_cli.py --explain <run_id>   # 单条因果链"
    )
