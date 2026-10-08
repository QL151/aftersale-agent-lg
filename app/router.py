"""意图路由层：决定这条工单"走确定性工作流"还是"交给自主 Agent"。

为什么要有这一层？（这是本项目最值得讲的设计）
  售后工单的意图分布是**极度长尾偏高**的：8 类高频意图占了 ~88% 的量，
  而它们每一个的处理步骤其实是**固定的**：
    「物流查询」= 拿订单 → 查轨迹 → 按 KB008 判断是否异常 → 回复
  这种任务用大模型自由发挥，有三个纯亏：
    ① 贵：每轮把 19 个工具描述 + 全部历史塞进 prompt
    ② 慢：多轮 ReAct 往返
    ③ **不稳**：模型可能忘记查政策就直接承诺退款
  所以：**能用确定性代码做的，就不要让模型做。**

三级路由（逐级兜底，不是单点判断）：
  ① 规则打分     —— 关键词/正则 + 槽位抽取，0 成本、0 延迟、可解释
  ② 模型分类     —— 规则打不出把握时，用小模型做一次意图分类（可选，可用开关关掉）
  ③ 长尾兜底     —— 仍无把握 → 进自主 Agent（ReAct），并标记 route="agent"

为什么规则层就够了？因为客服话术高度收敛：
  用户说"我的快递到哪了"和"物流怎么不动了"是同一个意图。
  规则层的召回在真实工单上通常能到 85%+，剩下的交给模型即可。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import settings

# ==================================================================== 意图定义


@dataclass
class IntentSpec:
    name: str                                  # 内部标识
    label: str                                 # 中文名（报表/工单用）
    workflow: str                              # 对应的确定性工作流
    patterns: list[tuple[str, float]]          # (正则, 权重)
    negative: list[str] = field(default_factory=list)   # 命中则降权（防止误判）
    tools: tuple[str, ...] = ()                # 工作流主要用到的工具（用于裁剪上下文）


INTENTS: list[IntentSpec] = [
    IntentSpec(
        name="logistics_query", label="物流查询", workflow="logistics_query",
        patterns=[
            (r"物流|快递|运单|包裹|发货", 1.0),
            (r"到哪|到哪了|什么时候(能|会)?到|多久(能|会)?到|还没到|没收到|没发货", 1.2),
            (r"签收|派送|揽收|转运|停滞|不动了", 1.0),
            (r"顺丰|中通|圆通|韵达|京东物流", 0.8),
        ],
        # 负向词是关键：中文里"发货"既可能是在问物流，也可能是在说"还没发货所以我要退款"。
        # 命中负向词就大幅降权，避免把退款诉求误判成物流查询。
        negative=[r"退款规则|价保|发票|优惠券", r"退款|退货|不要了|退了吧|退了",
                  r"分析|统计|占比|分布|异常率|排名"],
        tools=("get_order_detail", "get_logistics_trace", "search_knowledge_base"),
    ),
    IntentSpec(
        name="refund_progress", label="退款进度查询", workflow="refund_progress",
        patterns=[
            (r"退款.*(到账|进度|状态|多久|还没|查)|钱.*(退|到账)", 1.6),
            (r"退款到账|退款进度|退款查|退的钱|钱退到", 1.4),
            (r"退款", 0.5),
        ],
        negative=[r"我要退款|申请退款|想退款|帮我退"],
        tools=("get_order_detail", "search_knowledge_base"),
    ),
    IntentSpec(
        name="refund_request", label="退款申请", workflow="refund_request",
        patterns=[
            (r"我要退款|申请退款|想退款|帮我退|要求退款|退款吧|退给我", 1.8),
            (r"不想要了|不要了|退掉|退了", 1.2),
            (r"退款", 0.6),
        ],
        negative=[],
        tools=("get_order_detail", "check_refund_policy", "calc_refund_amount", "issue_refund"),
    ),
    IntentSpec(
        name="return_exchange", label="退货换货", workflow="return_exchange",
        patterns=[
            (r"退货|换货|退换|寄回|换一个|换个(尺码|颜色|规格)", 1.8),
            (r"尺码不对|大小不合适|穿着不合|颜色不对", 1.0),
        ],
        tools=("get_order_detail", "check_refund_policy", "search_knowledge_base"),
    ),
    IntentSpec(
        name="invoice_issue", label="发票问题", workflow="invoice_issue",
        patterns=[
            (r"发票|开票|抬头|税号|报销|专用发票|电子发票", 1.8),
        ],
        tools=("get_order_detail", "handle_invoice", "search_knowledge_base"),
    ),
    IntentSpec(
        name="price_protection", label="价保补差", workflow="price_protection",
        patterns=[
            (r"价保|保价|补差|降价|便宜了|差价|买贵", 1.8),
        ],
        tools=("get_order_detail", "check_price_protection", "search_knowledge_base"),
    ),
    IntentSpec(
        name="coupon_promo", label="优惠券活动咨询", workflow="coupon_promo",
        patterns=[
            (r"优惠券|券|满减|活动|折扣|补贴|红包", 1.5),
            (r"为什么不能用|用不了|失效|过期", 0.9),
        ],
        negative=[r"退款|物流|发票|价保"],
        tools=("get_customer_profile", "search_knowledge_base", "grant_coupon"),
    ),
    IntentSpec(
        name="complaint_escalation", label="投诉升级", workflow="complaint_escalation",
        patterns=[
            (r"投诉|举报|曝光|差评|315|消协|工商", 1.9),
            (r"太差|垃圾|欺骗|骗子|态度|不满|生气|气死|无语", 2.2),
            (r"找你们领导|要人工|转人工|人工客服|经理", 1.4),
            (r"赔偿|赔付|索赔", 1.2),
        ],
        tools=("get_order_detail", "get_logistics_trace", "get_customer_profile",
               "search_knowledge_base", "grant_coupon", "escalate_to_human"),
    ),
]

INTENT_BY_NAME = {item.name: item for item in INTENTS}
WORKFLOW_INTENTS = tuple(item.name for item in INTENTS)


# ==================================================================== 槽位抽取

_ORDER_RE = re.compile(r"\b(SO\d{6,})\b", re.IGNORECASE)
_CUSTOMER_ID_RE = re.compile(r"\b(C\d{5})\b", re.IGNORECASE)
_TRACKING_RE = re.compile(r"\b((?:SF|YT|ZT|JD)\d{10,})\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(\d+(?:\.\d{1,2})?)\s*(?:元|块|块钱|rmb)", re.IGNORECASE)
_CURRENT_PRICE_RE = re.compile(
    r"(?:现价|现在卖|现在(?:是|卖)|售价|降到|便宜到|只要)\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE
)
_PHONE_TAIL_RE = re.compile(r"尾号\s*(\d{4})")
_NAME_RE = re.compile(r"(?:我叫|我是|客户|姓名[:：]?)\s*([\u4e00-\u9fa5]{2,4})")


def extract_slots(text: str) -> dict[str, Any]:
    """从用户话里抽业务槽位。抽不到就让工作流去追问，绝不猜。"""
    slots: dict[str, Any] = {}
    orders = _ORDER_RE.findall(text)
    if orders:
        slots["order_id"] = orders[0].upper()
        if len({o.upper() for o in orders}) > 1:
            # 一句话里出现多个订单号 = 复合请求，交给自主 Agent 更合适
            slots["order_ids"] = [o.upper() for o in orders]
    if match := _CUSTOMER_ID_RE.search(text):
        slots["customer_id"] = match.group(1).upper()
    if match := _TRACKING_RE.search(text):
        slots["tracking_no"] = match.group(1).upper()
    if match := _CURRENT_PRICE_RE.search(text):
        slots["current_price"] = float(match.group(1))
    if match := _AMOUNT_RE.search(text):
        slots["amount"] = float(match.group(1))
    if match := _PHONE_TAIL_RE.search(text):
        slots["phone_tail"] = match.group(1)
    if match := _NAME_RE.search(text):
        slots["customer_name"] = match.group(1)
    if "紧急" in text or "急" in text:
        slots["priority"] = "紧急"
    return slots


# ==================================================================== 路由结果


@dataclass
class RouteDecision:
    intent: str
    label: str
    confidence: float
    route: str                      # workflow / agent / clarify
    workflow: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    source: str = "rule"            # rule / llm / fallback

    @property
    def is_workflow(self) -> bool:
        return self.route == "workflow"

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "route": self.route,
            "workflow": self.workflow,
            "slots": self.slots,
            "reason": self.reason,
            "source": self.source,
        }


# ==================================================================== 路由器


class IntentRouter:
    """三级路由。llm 只在需要兜底分类时才被调用。"""

    def __init__(self, llm: Any = None, threshold: float | None = None,
                 use_llm: bool | None = None) -> None:
        self.llm = llm
        self.threshold = threshold if threshold is not None else settings.router_confidence_threshold
        self.use_llm = settings.router_use_llm if use_llm is None else use_llm
        self.stats = {"rule": 0, "llm": 0, "agent": 0}

    # ------------------------------------------------------------ ① 规则打分
    def _score(self, text: str) -> tuple[dict[str, float], str]:
        scores: dict[str, float] = {}
        best_evidence = ""
        lowered = text.lower()
        for spec in INTENTS:
            score = 0.0
            evidence: list[str] = []
            for pattern, weight in spec.patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += weight
                    evidence.append(pattern)
            if score <= 0:
                continue
            for pattern in spec.negative:
                if re.search(pattern, text, re.IGNORECASE):
                    score *= 0.35          # 命中负向词 → 大幅降权（例如"退款规则"不是"我要退款"）
            scores[spec.name] = round(score, 3)
            if not best_evidence:
                best_evidence = ",".join(evidence)
        return scores, best_evidence

    @staticmethod
    def _confidence(scores: dict[str, float]) -> tuple[str, float]:
        if not scores:
            return "", 0.0
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_name, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = (top - second) / top if top else 0.0
        # 证据强度（top 分越高越有把握）× 与第二名的区分度
        strength = min(top / 2.2, 1.0)
        confidence = 0.35 + 0.5 * strength + 0.15 * margin
        return top_name, round(min(confidence, 0.97), 3)

    # ------------------------------------------------------------ ② 模型分类
    def _classify_with_llm(self, text: str) -> tuple[str, float] | None:
        if self.llm is None or not self.use_llm:
            return None
        options = "、".join(f"{item.name}({item.label})" for item in INTENTS)
        prompt = (
            "你是电商售后工单的意图分类器。请只输出一个 JSON，不要任何解释。\n"
            f"可选意图：{options}、long_tail(以上都不是)\n"
            '输出格式：{"intent": "...", "confidence": 0.0~1.0}\n\n'
            f"用户原话：{text}"
        )
        try:
            reply = self.llm.chat([{"role": "user", "content": prompt}], tools=None, temperature=0.0)
        except Exception:  # noqa: BLE001 —— 分类失败就当没分类，不能影响主流程
            return None
        match = re.search(r"\{.*\}", reply.text or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        name = str(data.get("intent", "")).strip()
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if name in INTENT_BY_NAME:
            return name, max(0.0, min(confidence, 1.0))
        return None

    # ------------------------------------------------------------ 主入口
    def route(self, text: str, slots: dict[str, Any] | None = None,
              memory_slots: dict[str, Any] | None = None) -> RouteDecision:
        text = (text or "").strip()
        memory_slots = memory_slots or {}
        merged_slots = {**memory_slots, **extract_slots(text), **(slots or {})}
        pending = memory_slots.get("_pending_intent") or ""

        # ---- 复合请求：一句话里出现多个订单号 → 交给自主 Agent 更合适 ----
        if merged_slots.get("order_ids"):
            self.stats["agent"] += 1
            return RouteDecision(
                intent="long_tail", label="复合请求", confidence=0.4, route="agent",
                slots=merged_slots,
                reason=f"检测到多个订单号 {merged_slots['order_ids']}，属于跨单复合请求，转自主 Agent",
                source="fallback",
            )

        scores, _evidence = self._score(text)
        best_name, confidence = self._confidence(scores)

        if best_name and confidence >= self.threshold:
            self.stats["rule"] += 1
            spec = INTENT_BY_NAME[best_name]
            if spec.name == "price_protection" and "current_price" not in merged_slots \
                    and "amount" in merged_slots:
                # 价保场景里用户说的"现在卖 343 元"就是现价，不是退款金额
                merged_slots["current_price"] = merged_slots["amount"]
            return RouteDecision(
                intent=spec.name, label=spec.label, confidence=confidence, route="workflow",
                workflow=spec.workflow, slots=merged_slots,
                reason=f"规则命中（得分 {scores.get(best_name)}，阈值 {self.threshold}）",
                scores=scores, source="rule",
            )

        # ---- 槽位追问的续接：上一轮问了"订单号是多少"，这一轮用户只回了订单号 ----
        # 如果不做这件事，多轮澄清就会丢意图 —— 这是"平均交互轮次"和完成率的主要杀手。
        if pending and pending in INTENT_BY_NAME:
            spec = INTENT_BY_NAME[pending]
            self.stats["rule"] += 1
            return RouteDecision(
                intent=spec.name, label=spec.label,
                confidence=max(confidence, 0.85), route="workflow",
                workflow=spec.workflow, slots=merged_slots,
                reason=f"沿用上一轮未完成的意图「{spec.label}」（本轮是槽位补充，非新意图）",
                scores=scores, source="pending_intent",
            )

        # 规则没把握 → 用小模型兜一次
        llm_result = self._classify_with_llm(text)
        if llm_result:
            name, llm_confidence = llm_result
            if llm_confidence >= self.threshold:
                self.stats["llm"] += 1
                spec = INTENT_BY_NAME[name]
                return RouteDecision(
                    intent=spec.name, label=spec.label, confidence=llm_confidence, route="workflow",
                    workflow=spec.workflow, slots=merged_slots,
                    reason=f"规则置信度不足（{confidence}），模型分类命中", scores=scores, source="llm",
                )

        # ③ 长尾 → 自主 Agent
        self.stats["agent"] += 1
        return RouteDecision(
            intent=best_name or "long_tail",
            label=INTENT_BY_NAME[best_name].label if best_name else "长尾咨询",
            confidence=confidence, route="agent", workflow="",
            slots=merged_slots,
            reason=(
                f"无确定性工作流覆盖（规则最高置信度 {confidence} < 阈值 {self.threshold}），"
                "转自主 Agent 处理"
            ),
            scores=scores, source="fallback",
        )


def describe_routing() -> str:
    lines = ["意图路由表（8 类高频意图 → 确定性工作流，其余 → 自主 Agent）", "=" * 88]
    for index, spec in enumerate(INTENTS, 1):
        lines.append(f"  {index}. {spec.label:<16} intent={spec.name:<22} workflow={spec.workflow}")
        lines.append(f"     工具：{'、'.join(spec.tools)}")
        lines.append(f"     触发：{' / '.join(p for p, _ in spec.patterns[:3])}")
    lines.append("")
    lines.append("  9. 长尾（其它一切）  →  自主 Agent（ReAct + Function Calling，全量工具）")
    lines.append("=" * 88)
    lines.append(f"置信度阈值：{settings.router_confidence_threshold}（低于它判定为长尾）")
    return "\n".join(lines)
