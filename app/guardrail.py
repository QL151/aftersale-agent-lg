"""Guardrail：Agent 的安全边界。

为什么 Agent 比普通程序更需要它？
  普通程序的执行路径是你写死的；Agent 的路径是**模型临时决定的**。
  模型可能被提示注入、可能理解错任务、可能陷入循环 ——
  所以必须在"模型想做"和"真的去做"之间插一层独立检查。

本项目的四层防护（比通用 Agent 多了一层"业务规则"，这是关键差异）：

  ① 参数级     类型 / 必填 / 范围（由 tools/base.py 的 validate 完成）
  ② 内容级     代码沙箱 AST 校验、只读 SQL 策略
  ③ **业务级** ★ 退款金额不得超过测算值；发券不得超过 100 元
                —— 这一层通用框架没有，必须自己写。
                因为最贵的错误不是"Agent 崩溃"，而是"Agent 给某个客户多退了 800 块"。
  ④ 行为级     调用预算 / 重复调用检测 / 总时长 / 关键动作人工确认

核心设计：**被拦截不是"崩溃"，而是把拒绝原因回传给模型**，
让模型有机会换个安全做法 —— 这叫"可恢复的护栏"。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, settings as global_settings
from .tools.base import Tool

_FORBIDDEN_SQL = re.compile(
    r"\b(drop|delete|update|insert|alter|create|attach|detach|pragma|vacuum|replace)\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
# 提示注入的典型话术：出现即降级为"需人工确认"
_INJECTION = re.compile(
    r"(忽略(以上|之前|上面).{0,6}(指令|要求|规则)|ignore\s+(all\s+)?previous|"
    r"你现在是|扮演.{0,4}管理员|system\s*[:：]|直接给我退|不用审核|跳过审批|"
    r"把(退款)?金额改成|全额退给我)",
    re.IGNORECASE,
)

# 业务硬约束（写死在代码里，不通过配置放开 —— 这类约束不该被"配置错误"绕过）
MAX_COUPON = 100.0
REFUND_TOLERANCE = 0.01


@dataclass
class Decision:
    allow: bool
    reason: str = ""
    needs_confirm: bool = False
    policy: str = ""
    risk_level: str = "low"


@dataclass
class RunContext:
    """一次运行的运行时状态（行为级预算靠它）。"""

    started_at: float = field(default_factory=time.time)
    tool_calls: int = 0
    blocked: int = 0
    confirmations: int = 0
    call_counter: dict[str, int] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at


class Guardrail:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or global_settings

    # ------------------------------------------------------------- 核心检查
    def check(self, tool: Tool, arguments: dict[str, Any], state: Any = None,
              ctx: RunContext | None = None) -> Decision:
        ctx = ctx or RunContext()

        # ---------- ④ 行为级：预算 ----------
        if ctx.tool_calls >= self.s.max_tool_calls:
            return Decision(False, f"本次任务最多调用 {self.s.max_tool_calls} 次工具，已达上限",
                            policy="budget")
        if ctx.elapsed > self.s.max_total_seconds:
            return Decision(False, f"任务总时长超过 {self.s.max_total_seconds:.0f} 秒上限",
                            policy="budget")

        # ---------- ① 参数级 ----------
        error = tool.validate(arguments)
        if error:
            return Decision(False, f"参数不合法：{error}", policy="schema")

        # ---------- ④ 行为级：重复调用 ----------
        signature = f"{tool.name}:{sorted(arguments.items())}"
        count = ctx.call_counter.get(signature, 0)
        if count >= self.s.max_same_call and not tool.idempotent:
            return Decision(
                False,
                f"同一个工具用完全相同的参数已调用 {count} 次，请换参数或换思路",
                policy="repeat",
            )

        # ---------- ② 内容级：代码沙箱 ----------
        if tool.policy == "sandbox":
            from .sandbox import validate_code

            reason = validate_code(str(arguments.get("code", "")))
            if reason:
                return Decision(False, f"沙箱静态校验未通过：{reason}", policy="sandbox")
            return Decision(True, "已通过沙箱静态校验（无需人工确认）", policy="sandbox",
                            risk_level=tool.risk_level)

        # ---------- ② 内容级：只读 SQL ----------
        if tool.policy == "readonly_sql":
            sql = str(arguments.get("sql", "")).strip().rstrip(";")
            if not sql.lower().startswith("select"):
                return Decision(False, "该工具只允许 SELECT 查询", policy="readonly_sql")
            if _FORBIDDEN_SQL.search(sql):
                return Decision(False, f"检测到写操作关键字，拒绝执行：{sql[:60]}",
                                policy="readonly_sql")
            if ";" in sql:
                return Decision(False, "不允许多条语句", policy="readonly_sql")
            return Decision(True, "已通过只读 SQL 策略", policy="readonly_sql")

        # ---------- ③ 业务级：退款金额不得超过测算值 ----------
        if tool.name == "issue_refund":
            verdict = self._check_refund_amount(arguments, state)
            if not verdict.allow:
                return verdict

        # ---------- ③ 业务级：发券上限 ----------
        if tool.name == "grant_coupon":
            amount = float(arguments.get("amount", 0) or 0)
            if amount > MAX_COUPON:
                return Decision(
                    False,
                    f"补偿券金额 {amount} 元超过业务硬上限 {MAX_COUPON} 元，请人工走特殊审批",
                    policy="business_rule",
                )

        # ---------- ② 内容级：提示注入检测 ----------
        injected = self._detect_injection(state)

        # ---------- ④ 行为级：关键动作人工确认 ----------
        if tool.dangerous:
            if self.s.confirm_mode == "allow":
                return Decision(True, "关键动作（配置为自动放行）", policy="auto_allow",
                                risk_level=tool.risk_level)
            if self.s.confirm_mode == "deny":
                return Decision(False, "关键动作（配置为自动拒绝）", policy="auto_deny")
            reason = "关键动作需要人工审批"
            if injected:
                reason += f"；且检测到疑似提示注入（{injected}），一律人工复核"
            return Decision(True, reason, needs_confirm=True, policy="hitl",
                            risk_level=tool.risk_level)

        if injected:
            return Decision(
                True, f"检测到疑似提示注入（{injected}），本轮结论需人工复核",
                needs_confirm=False, policy="injection_warn", risk_level=tool.risk_level,
            )

        return Decision(True, risk_level=tool.risk_level)

    # ------------------------------------------------------------- 业务规则
    def _check_refund_amount(self, arguments: dict, state: Any) -> Decision:
        """退款金额校验 —— 本项目最重要的一条业务护栏。

        为什么必须有？
          模型在长对话里很容易把"应退 1708.1 元"写成"退 1798 元"（把运费也算进去），
          或者被用户诱导"全额退给我"。金额是**不可逆**的，所以：
            必须存在测算记录（calc_refund_amount 的产物），且申请金额不得超过它。
          这不是"不信任模型"，是"不让模型的失误直接变成资金损失"。
        """
        quote = None
        if state is not None:
            facts = getattr(state, "facts", {}) or {}
            quote = facts.get("refund_quote")
        if quote is None:
            return Decision(
                False,
                "缺少退款测算依据：必须先调用 calc_refund_amount 得到应退金额，再发起退款",
                policy="business_rule",
            )
        allowed = float(quote.get("total", 0.0)) if isinstance(quote, dict) else float(quote)
        requested = float(arguments.get("amount", 0) or 0)
        if requested > allowed + REFUND_TOLERANCE:
            return Decision(
                False,
                f"申请退款 {requested} 元超过测算应退金额 {allowed} 元，已拒绝。"
                "如确有特殊情况，请转人工审批",
                policy="business_rule",
            )
        return Decision(True, f"退款金额 {requested} 元在测算范围内（上限 {allowed} 元）",
                        policy="business_rule")

    @staticmethod
    def detect_injection(state: Any) -> str:
        """公开的提示注入检测：工作流的风控节点也会用它强制走人工。"""
        if state is None:
            return ""
        task = getattr(state, "task", "") or ""
        match = _INJECTION.search(task)
        return match.group(0)[:30] if match else ""

    @staticmethod
    def _detect_injection(state: Any) -> str:
        return Guardrail.detect_injection(state)

    # ------------------------------------------------------------- 输出脱敏
    @staticmethod
    def sanitize(text: str) -> str:
        """敏感信息脱敏：进 Trace 和进模型上下文之前都该做。"""
        if not text:
            return text
        return _PHONE.sub(r"\1****\3", _EMAIL.sub("***@***", text))

    @staticmethod
    def describe_block(tool: Tool, decision: Decision) -> str:
        return f"工具 {tool.name} 被拦截：{decision.reason}"
