"""核心数据结构。

分三层：
  ① 领域实体（Order / Shipment / Ticket / Customer ...）—— 业务数据长什么样
  ② 运行时对象（ToolCall / ToolResult / Step / RunResult）—— 一次任务怎么被记录
  ③ 图状态（AgentState）—— 状态机在节点之间传递的那份"全局状态"

为什么用 dataclass 而不是 Pydantic？
  这些是**内部运行时对象**，不需要反序列化校验，dataclass 更轻更直白。
  对外 HTTP 接口的请求/响应模型才用 Pydantic（见 app/api/models.py）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# =============================================================== ① 领域实体


@dataclass
class Order:
    order_id: str
    customer_id: str
    customer_name: str
    contact: str
    sku_name: str
    category: str
    amount: float            # 实付金额
    shipping_fee: float
    paid_at: str
    delivered_at: str        # 空串 = 未签收
    status: str              # 已支付 / 已发货 / 已签收 / 已退款 / 已完成 / 已取消
    invoice_status: str      # 未开票 / 已开票 / 开票中
    price_protect_days: int  # 价保天数
    is_opened: int           # 是否拆封：1/0

    def public(self) -> dict:
        """给模型看的版本：联系方式脱敏。"""
        return {
            "order_id": self.order_id,
            "customer_name": self.customer_name,
            "sku_name": self.sku_name,
            "category": self.category,
            "amount": self.amount,
            "shipping_fee": self.shipping_fee,
            "paid_at": self.paid_at,
            "delivered_at": self.delivered_at or "未签收",
            "status": self.status,
            "invoice_status": self.invoice_status,
        }


@dataclass
class ShipmentEvent:
    time: str
    status: str
    location: str
    description: str


@dataclass
class Shipment:
    tracking_no: str
    carrier: str
    order_id: str
    status: str                        # 已揽收 / 运输中 / 派送中 / 已签收 / 异常
    events: list[ShipmentEvent] = field(default_factory=list)

    @property
    def last_event(self) -> ShipmentEvent | None:
        return self.events[-1] if self.events else None


@dataclass
class PolicyVerdict:
    """退款/退货政策校验结论 —— 必须带条款依据（可解释性）。"""

    allowed: bool
    reason: str
    clause: str = ""                # 引用的条款编号，例如 "售后规则 3.2"
    needs_manual: bool = False      # 规则覆盖不到的灰区 → 转人工
    max_refund: float = 0.0


@dataclass
class RefundQuote:
    """退款金额计算结果 —— 每一项都要能解释。"""

    order_id: str
    goods_amount: float
    shipping_refund: float
    promo_deduct: float
    depreciation: float
    total: float
    breakdown: list[str] = field(default_factory=list)


@dataclass
class Customer:
    customer_id: str
    name: str
    phone: str
    level: str                 # 普通 / 银牌 / 金牌 / 黑名单观察
    registered_at: str
    refund_count_90d: int
    complaint_count_90d: int
    risk_score: float          # 0~1，越高越需要人工复核

    def public(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "level": self.level,
            "refund_count_90d": self.refund_count_90d,
            "complaint_count_90d": self.complaint_count_90d,
            "risk_score": self.risk_score,
        }


@dataclass
class Ticket:
    ticket_id: str
    order_id: str
    customer_id: str
    intent: str
    title: str
    status: str                # 待处理 / 处理中 / 待人工 / 已关闭
    assignee: str              # "AI" 表示 Agent 自动闭环
    created_at: str
    updated_at: str
    closed_at: str = ""
    auto_closed: int = 0       # 1 = 全自动闭环（无人介入）

    def public(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "order_id": self.order_id,
            "intent": self.intent,
            "title": self.title,
            "status": self.status,
            "assignee": self.assignee,
            "auto_closed": bool(self.auto_closed),
        }


# =============================================================== ② 运行时对象


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    model: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: str = ""
    error: str = ""
    blocked: bool = False
    block_reason: str = ""
    duration_ms: int = 0
    confirmed: bool | None = None
    requires_approval: bool = False     # 被 HITL 挂起等待人工
    approval_id: str = ""

    def to_llm_content(self) -> str:
        """回传给模型的内容。被拦下时也要回传 —— 模型才有机会换个安全做法。"""
        if self.requires_approval:
            return (
                f"[等待人工审批] 工具 {self.name} 属于关键动作，已提交人工审批"
                f"（审批单 {self.approval_id}），尚未执行。请先告知用户正在等待审核。"
            )
        if self.blocked:
            return (
                f"[已拦截] 工具 {self.name} 未执行。原因：{self.block_reason}。"
                f"请换一种安全的方式，或向用户说明为什么做不到。"
            )
        if not self.ok:
            return f"[执行失败] 工具 {self.name} 报错：{self.error}。请调整参数或换别的工具。"
        return f"[执行成功] {self.name} 返回：\n{self.output}"


@dataclass
class Step:
    """Trace 里的一步。

    type 取值：route / node / llm_call / tool_call / guardrail / approval / cost / final / error
    """

    index: int
    type: str
    duration_ms: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"index": self.index, "type": self.type, "duration_ms": self.duration_ms, **self.detail}


@dataclass
class RunResult:
    run_id: str
    thread_id: str
    task: str
    answer: str
    intent: str = ""
    route: str = ""                    # workflow / agent / clarify
    workflow: str = ""
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = "finished"   # finished / max_steps / timeout / error / interrupted / denied
    total_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_yuan: float = 0.0
    model_used: str = ""
    downgraded: bool = False
    handover: bool = False             # 是否转人工
    auto_closed: bool = False          # 是否全自动闭环（无人介入）
    approvals: int = 0
    engine: str = ""                   # native / langgraph
    turns: int = 1                     # 交互轮次（含澄清追问）

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.type == "tool_call")

    @property
    def blocked_calls(self) -> int:
        return sum(1 for s in self.steps if s.type == "guardrail" and s.detail.get("blocked"))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "task": self.task,
            "answer": self.answer,
            "intent": self.intent,
            "route": self.route,
            "workflow": self.workflow,
            "stopped_reason": self.stopped_reason,
            "total_ms": self.total_ms,
            "tool_calls": self.tool_calls,
            "blocked_calls": self.blocked_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_yuan": round(self.cost_yuan, 6),
            "model_used": self.model_used,
            "downgraded": self.downgraded,
            "handover": self.handover,
            "auto_closed": self.auto_closed,
            "approvals": self.approvals,
            "engine": self.engine,
            "turns": self.turns,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# =============================================================== ③ 图状态


@dataclass
class AgentState:
    """状态机在节点间传递的全局状态。

    为什么把状态收敛成一个对象？
      状态机最怕"状态散落在闭包里"。所有节点只读写这一份状态，
      于是 checkpoint（中断恢复）和 trace（回放）都变成"存/取一个对象"。
    """

    run_id: str
    thread_id: str
    task: str
    session_id: str = ""

    # 路由结果
    intent: str = ""
    route: str = ""
    workflow: str = ""
    confidence: float = 0.0
    slots: dict[str, Any] = field(default_factory=dict)

    # 执行过程
    messages: list[dict] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)   # 工具查到的事实
    steps: list[Step] = field(default_factory=list)
    tool_calls: int = 0
    blocked: int = 0
    call_counter: dict[str, int] = field(default_factory=dict)

    # 人机协同
    pending_approval: dict | None = None
    approvals: int = 0
    approval_denied: bool = False

    # 成本
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_yuan: float = 0.0
    model_used: str = ""
    downgraded: bool = False

    # 结论
    answer: str = ""
    needs_clarify: bool = False
    clarify_question: str = ""
    handover: bool = False
    auto_closed: bool = False
    turns: int = 1
    retries: dict[str, int] = field(default_factory=dict)

    def fact(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def put(self, key: str, value: Any) -> None:
        self.facts[key] = value

    # ---------------------------------------------------------- 存档 / 恢复
    # 状态机要能"中断 → 落盘 → 进程重启 → 接着跑"，所以状态必须可序列化。
    # 手写 to_payload/from_payload 而不是 dataclasses.asdict：
    #   嵌套的 Step 需要还原成对象，set/dict 也需要安全转换。
    def to_payload(self) -> dict:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "task": self.task,
            "session_id": self.session_id,
            "intent": self.intent,
            "route": self.route,
            "workflow": self.workflow,
            "confidence": self.confidence,
            "slots": _jsonable(self.slots),
            "messages": _jsonable(self.messages),
            "facts": _jsonable(self.facts),
            "steps": [s.to_dict() for s in self.steps],
            "tool_calls": self.tool_calls,
            "blocked": self.blocked,
            "call_counter": _jsonable(self.call_counter),
            "pending_approval": _jsonable(self.pending_approval),
            "approvals": self.approvals,
            "approval_denied": self.approval_denied,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_yuan": self.cost_yuan,
            "model_used": self.model_used,
            "downgraded": self.downgraded,
            "answer": self.answer,
            "needs_clarify": self.needs_clarify,
            "clarify_question": self.clarify_question,
            "handover": self.handover,
            "auto_closed": self.auto_closed,
            "turns": self.turns,
            "retries": _jsonable(self.retries),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "AgentState":
        state = cls(
            run_id=payload.get("run_id", ""),
            thread_id=payload.get("thread_id", ""),
            task=payload.get("task", ""),
        )
        for key, value in payload.items():
            if key == "steps":
                state.steps = [
                    Step(
                        index=s.get("index", 0),
                        type=s.get("type", ""),
                        duration_ms=s.get("duration_ms", 0),
                        detail={k: v for k, v in s.items()
                                if k not in {"index", "type", "duration_ms"}},
                    )
                    for s in value
                ]
            elif hasattr(state, key):
                setattr(state, key, value)
        return state


def _jsonable(value: Any) -> Any:
    """把状态里的值转成可 JSON 序列化的形式（存档用）。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return repr(value)
