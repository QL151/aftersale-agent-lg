"""售后业务工具集 —— Agent 的"手"。

一共 19 个工具，分成三档：
  · 12 个**业务工具**：订单/客户、物流、退款规则与计算、价保、发票、工单流转、知识库 —— 只读或低副作用
  ·  2 个**关键动作**：issue_refund（退款）、grant_coupon（发券）—— 会动钱，一律走人机协同（HITL）
  ·  5 个**辅助工具**：add_ticket_note / send_notification / get_current_time / calculator / run_analysis_code

每个工具的描述都是写给模型看的"什么时候该用它"，不是写给程序员看的注释 ——
工具描述是 Agent 效果的第一生产力。
"""

from __future__ import annotations

import json

from .. import clock
from ..db import Repo, calc_refund, check_policy
from ..schemas import Customer, Order
from .base import ToolContext, tool

# ==================================================================== 内部工具


def _repo(ctx: ToolContext | None) -> Repo:
    if ctx is not None and ctx.repo is not None:
        return ctx.repo
    return Repo()


def _fmt_order(order: Order) -> str:
    lines = [
        f"订单号：{order.order_id}",
        f"客户：{order.customer_name}（{order.customer_id}）",
        f"商品：{order.sku_name}（类目：{order.category}）",
        f"实付：{order.amount} 元，运费：{order.shipping_fee} 元",
        f"下单时间：{order.paid_at}",
        f"签收时间：{order.delivered_at or '未签收'}",
        f"订单状态：{order.status}",
        f"发票状态：{order.invoice_status}",
        f"是否已拆封：{'是' if order.is_opened else '否'}",
    ]
    return "\n".join(lines)


def _resolve_order(repo: Repo, order_id: str = "", customer_name: str = "") -> tuple[Order | None, str]:
    """把"订单号/客户名"解析成唯一订单。多义时明确返回候选，让 Agent 去追问。"""
    if order_id:
        order = repo.get_order(order_id)
        if order is None:
            return None, f"没有找到订单号 {order_id}，请确认后重试。"
        return order, ""
    if not customer_name:
        return None, "需要提供订单号，或提供客户姓名以便查询。"

    customers = repo.find_customers_by_name(customer_name)
    if not customers:
        return None, f"没有找到客户「{customer_name}」。"
    if len(customers) > 1:
        names = "、".join(f"{c.name}({c.customer_id})" for c in customers[:5])
        return None, f"客户名「{customer_name}」匹配到多个账号：{names}。请让用户提供订单号或注册手机号后四位。"

    orders = repo.list_orders_by_customer(customers[0].customer_id)
    if not orders:
        return None, f"客户「{customer_name}」名下没有订单。"
    if len(orders) > 1:
        brief = "、".join(f"{o.order_id}({o.sku_name},{o.status})" for o in orders[:5])
        return None, f"客户「{customer_name}」有多个订单：{brief}。请让用户指明具体订单号。"
    return orders[0], ""


# ==================================================================== 12 个业务工具


@tool(
    name="get_order_detail",
    description=(
        "根据订单号查询订单详情（商品、金额、运费、下单/签收时间、订单状态、发票状态、是否拆封）。"
        "这是绝大多数售后问题的第一步：先拿到订单事实，再判断能不能退、退多少。"
        "如果不确定订单号，先用 search_orders_by_customer。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 4, "maxLength": 24, "description": "订单号，例如 SO2026000123"}
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    tags=("order", "refund", "logistics", "invoice", "price", "ticket"),
)
def get_order_detail(order_id: str, _ctx: ToolContext | None = None) -> str:
    order = _repo(_ctx).get_order(order_id)
    if order is None:
        return f"没有找到订单 {order_id}。请确认订单号是否正确。"
    return _fmt_order(order)


@tool(
    name="search_orders_by_customer",
    description=(
        "按客户姓名查询其名下最近的订单列表（只读）。"
        "当用户只说姓名和商品、没有给订单号时使用，用来定位到具体订单号。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "customer_name": {"type": "string", "minLength": 1, "maxLength": 20, "description": "客户姓名"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "最多返回几条，默认 5"},
        },
        "required": ["customer_name"],
        "additionalProperties": False,
    },
    tags=("order", "refund", "logistics", "invoice", "price", "ticket"),
)
def search_orders_by_customer(
    customer_name: str, limit: int = 5, _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    customers = repo.find_customers_by_name(customer_name)
    if not customers:
        return f"没有找到客户「{customer_name}」。"
    chunks = []
    for customer in customers[:3]:
        orders = repo.list_orders_by_customer(customer.customer_id, limit=limit)
        head = f"客户 {customer.name}（{customer.customer_id}，{customer.level}）名下 {len(orders)} 单："
        body = "\n".join(
            f"  {o.order_id} | {o.sku_name} | {o.amount} 元 | {o.status} | 签收 {o.delivered_at or '未签收'}"
            for o in orders
        )
        chunks.append(head + ("\n" + body if body else "（无订单）"))
    return "\n".join(chunks)


@tool(
    name="get_customer_profile",
    description=(
        "查询客户画像：会员等级、近 90 天退款/投诉次数、风险评分，以及系统记录的长期服务偏好。"
        "在决定是否给补偿、是否需要人工复核时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "minLength": 3, "maxLength": 20, "description": "客户号，例如 C00001"},
            "customer_name": {"type": "string", "minLength": 1, "maxLength": 20, "description": "客户姓名（不知道客户号时用）"},
        },
        "additionalProperties": False,
    },
    tags=("order", "refund", "complaint", "coupon", "ticket"),
)
def get_customer_profile(
    customer_id: str = "", customer_name: str = "", _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    customer: Customer | None = None
    if customer_id:
        customer = repo.get_customer(customer_id)
    elif customer_name:
        found = repo.find_customers_by_name(customer_name)
        if len(found) == 1:
            customer = found[0]
        elif len(found) > 1:
            return "匹配到多个客户：" + "、".join(f"{c.name}({c.customer_id})" for c in found[:5])
    if customer is None:
        return "没有找到该客户，请确认客户号或姓名。"

    lines = [
        f"客户：{customer.name}（{customer.customer_id}）",
        f"会员等级：{customer.level}",
        f"注册时间：{customer.registered_at}",
        f"近 90 天退款次数：{customer.refund_count_90d}",
        f"近 90 天投诉次数：{customer.complaint_count_90d}",
        f"风险评分：{customer.risk_score}（越高越建议人工复核）",
    ]
    # 长期偏好来自 PostgreSQL（不可用时是 SQLite 降级实现）
    if _ctx is not None and _ctx.profile is not None:
        prefs = _ctx.profile.all_preferences(customer.customer_id)
        if prefs:
            lines.append("长期服务偏好：" + json.dumps(prefs, ensure_ascii=False))
    return "\n".join(lines)


@tool(
    name="get_logistics_trace",
    description=(
        "查询物流轨迹（承运商、运单号、当前状态、完整轨迹时间线）。"
        "用户问「到哪了」「为什么还没到」时必须用它拿真实轨迹，不要凭空推测。"
        "订单号或运单号给出一个即可。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 4, "maxLength": 24, "description": "订单号"},
            "tracking_no": {"type": "string", "minLength": 6, "maxLength": 24, "description": "运单号"},
        },
        "additionalProperties": False,
    },
    tags=("logistics", "order", "complaint", "refund"),
)
def get_logistics_trace(
    order_id: str = "", tracking_no: str = "", _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    shipment = (
        repo.get_shipment_by_order(order_id) if order_id else repo.get_shipment_by_tracking(tracking_no)
    )
    if shipment is None:
        target = order_id or tracking_no
        return f"没有查到 {target} 的物流信息。可能是商家尚未发货。"

    lines = [
        f"运单号：{shipment.tracking_no}",
        f"承运商：{shipment.carrier}",
        f"当前状态：{shipment.status}",
        "轨迹：",
    ]
    lines += [f"  {e.time} | {e.status} | {e.location} | {e.description}" for e in shipment.events]

    if shipment.status == "异常":
        stale_days = clock.days_since(shipment.last_event.time) if shipment.last_event else None
        lines.append(
            f"⚠ 异常提示：物流已停滞约 {stale_days:.1f} 天。"
            "按售后规则 KB008，超过 72 小时无更新即视为异常，应先向承运商发起查询，"
            "48 小时未恢复可按丢件处理（补发或全额退款）。"
            if stale_days is not None
            else "⚠ 该运单已被标记为异常。"
        )
    return "\n".join(lines)


@tool(
    name="check_refund_policy",
    description=(
        "校验某订单是否符合退款/退货政策，返回是否可退、原因、以及**引用的规则条款号**和最高可退金额。"
        "任何退款类结论都必须先调用它拿到条款依据，禁止凭经验判断是否可退。"
        "quality_issue=true 表示用户主张质量问题（走质量问题通道，时效更长但可能需凭证）。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 4, "maxLength": 24, "description": "订单号"},
            "quality_issue": {"type": "boolean", "description": "是否主张质量问题，默认 false"},
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    tags=("refund", "return", "order", "complaint"),
)
def check_refund_policy(
    order_id: str, quality_issue: bool = False, _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    order = repo.get_order(order_id)
    if order is None:
        return f"没有找到订单 {order_id}。"
    rule = repo.get_rule(order.category)
    verdict = check_policy(order, rule, quality_issue=quality_issue)
    lines = [
        f"订单 {order_id}（{order.category}）政策校验结果：",
        f"  是否可退：{'可以' if verdict.allowed else '不可以'}",
        f"  依据条款：{verdict.clause or '无'}",
        f"  说明：{verdict.reason}",
    ]
    if verdict.max_refund:
        lines.append(f"  最高可退：{verdict.max_refund} 元")
    if verdict.needs_manual:
        lines.append("  需要人工复核：是（该场景规则未完全覆盖，请转人工处理，不要自行承诺退款）")
    if rule and rule.get("note"):
        lines.append(f"  类目备注：{rule['note']}")
    return "\n".join(lines)


@tool(
    name="calc_refund_amount",
    description=(
        "计算某订单的实际退款金额，并给出逐项明细（商品实付、运费退还、优惠分摊扣回、折旧）。"
        "确定可退之后、执行退款之前调用它拿到准确金额。金额一律以本工具的返回为准，不要自己算。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 4, "maxLength": 24, "description": "订单号"},
            "quality_issue": {"type": "boolean", "description": "是否质量问题（影响运费与折旧）"},
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    tags=("refund", "return", "order"),
)
def calc_refund_amount(
    order_id: str, quality_issue: bool = False, _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    order = repo.get_order(order_id)
    if order is None:
        return f"没有找到订单 {order_id}。"
    rule = repo.get_rule(order.category)
    quote = calc_refund(order, repo.promo_deduct(order_id), rule, quality_issue=quality_issue)
    lines = [f"订单 {order_id} 退款测算：", *[f"  · {b}" for b in quote.breakdown]]
    lines.append(f"结论：应退客户 {quote.total} 元。")
    return "\n".join(lines)


@tool(
    name="check_price_protection",
    description=(
        "价保校验：判断订单是否在价保期内、是否满足补差条件，并算出可补差价。"
        "用户说「刚买就降价了」「能补差价吗」时使用。需要用户提供当前售价。"
        "注意：秒杀/拼团/优惠券造成的价差不参与价保。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 4, "maxLength": 24, "description": "订单号"},
            "current_price": {"type": "number", "minimum": 0.01, "maximum": 100000, "description": "当前同款商品官方售价（元）"},
            "is_promo": {"type": "boolean", "description": "该低价是否来自秒杀/拼团/限时活动，默认 false"},
        },
        "required": ["order_id", "current_price"],
        "additionalProperties": False,
    },
    tags=("price", "order"),
)
def check_price_protection(
    order_id: str, current_price: float, is_promo: bool = False, _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    order = repo.get_order(order_id)
    if order is None:
        return f"没有找到订单 {order_id}。"

    lines = [f"订单 {order_id} 价保校验：", f"  实付金额：{order.amount} 元；现价：{current_price} 元"]

    if is_promo:
        return "\n".join(
            lines
            + ["  结论：不满足价保条件。", "  依据：售后规则 KB005 —— 秒杀/拼团/限时优惠造成的价差不参与价保。"]
        )
    if not order.delivered_at:
        return "\n".join(
            lines + ["  结论：订单尚未签收，价保自签收之日起算，暂不能申请。", "  依据：KB005。"]
        )
    elapsed = clock.days_since(order.delivered_at) or 0
    if elapsed > order.price_protect_days:
        return "\n".join(
            lines
            + [
                f"  结论：已签收 {elapsed:.1f} 天，超过 {order.price_protect_days} 天价保期，不能申请。",
                "  依据：KB005 价格保护规则。",
            ]
        )
    diff = round(order.amount - current_price, 2)
    if diff <= 0:
        return "\n".join(
            lines + ["  结论：当前售价不低于实付金额，无差价可补。", "  依据：KB005。"]
        )
    return "\n".join(
        lines
        + [
            f"  结论：满足价保条件，可补差价 {diff} 元（签收后 {elapsed:.1f} 天，在 {order.price_protect_days} 天内）。",
            "  依据：KB005 价格保护规则。每个订单仅可申请一次。",
        ]
    )


@tool(
    name="handle_invoice",
    description=(
        "处理发票相关请求：查询开票状态、申请开票、换开发票、催办。"
        "用户提到「发票」「开票」「报销」「抬头」时使用。"
        "修改发票抬头等动作会留下操作记录。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 4, "maxLength": 24, "description": "订单号"},
            "action": {
                "type": "string",
                "enum": ["查询", "申请开票", "换开", "催办"],
                "description": "要执行的发票动作",
            },
            "title_type": {
                "type": "string",
                "enum": ["个人", "企业"],
                "description": "抬头类型，申请开票/换开时需要",
            },
        },
        "required": ["order_id", "action"],
        "additionalProperties": False,
    },
    tags=("invoice", "order"),
)
def handle_invoice(
    order_id: str, action: str, title_type: str = "个人", _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    order = repo.get_order(order_id)
    if order is None:
        return f"没有找到订单 {order_id}。"

    if action == "查询":
        return (
            f"订单 {order_id} 发票状态：{order.invoice_status}。\n"
            "依据 KB004：订单完成后 30 天内可申请电子发票；企业抬头需提供公司名称与税号。"
        )
    if action == "申请开票":
        if order.invoice_status == "已开票":
            return f"订单 {order_id} 已开票，请勿重复申请。如需变更抬头请使用「换开」。"
        if order.status == "已取消":
            return f"订单 {order_id} 已取消，无法开票。"
        repo.set_invoice_status(order_id, "开票中")
        repo.log_event(_ctx.run_id if _ctx else "", "invoice", f"{order_id} 申请开票({title_type})")
        return (
            f"已为订单 {order_id} 提交{title_type}抬头开票申请，当前状态：开票中。\n"
            "依据 KB004：开票处理时效 1-3 个工作日，电子发票将发送至下单预留手机号。"
        )
    if action == "换开":
        if order.invoice_status != "已开票":
            return f"订单 {order_id} 当前状态为「{order.invoice_status}」，没有可换开的发票。"
        repo.set_invoice_status(order_id, "开票中")
        return (
            f"已为订单 {order_id} 提交换开申请（新抬头类型：{title_type}），原发票将作废。\n"
            "依据 KB004：换开需先作废原发票，处理时效 1-3 个工作日。"
        )
    # 催办
    return (
        f"已为订单 {order_id} 提交发票催办（当前状态：{order.invoice_status}），"
        "已通知开票专员优先处理，预计 1 个工作日内反馈。"
    )


@tool(
    name="create_ticket",
    description=(
        "创建一个售后工单，返回工单号。"
        "凡是无法当场闭环、需要后续跟进、或需要人工介入的售后请求，都应该建单留痕。"
        "title 要能一句话说清问题，便于人工接手。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 4, "maxLength": 24, "description": "关联订单号"},
            "intent": {"type": "string", "minLength": 2, "maxLength": 40, "description": "售后意图，例如 退款申请 / 物流查询"},
            "title": {"type": "string", "minLength": 4, "maxLength": 100, "description": "工单标题"},
        },
        "required": ["order_id", "intent", "title"],
        "additionalProperties": False,
    },
    tags=("ticket", "order", "complaint", "refund", "logistics", "invoice"),
)
def create_ticket(
    order_id: str, intent: str, title: str, _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    order = repo.get_order(order_id)
    if order is None:
        return f"没有找到订单 {order_id}，无法建单。"
    ticket = repo.create_ticket(order_id, order.customer_id, intent, title)
    if _ctx is not None:
        _ctx.extra["ticket_id"] = ticket.ticket_id
    return (
        f"已创建工单 {ticket.ticket_id}（状态：{ticket.status}，处理人：AI）。\n"
        f"标题：{ticket.title}\n关联订单：{order_id}"
    )


@tool(
    name="escalate_to_human",
    description=(
        "把工单升级给人工专员，并把工单状态置为「待人工」。"
        "遇到以下情况必须用它：规则判定需要人工复核、客户明确要求人工、投诉/情绪激烈、"
        "你的工具做不到的事。升级时要说清已经查清了什么、还差什么，人工接手才快。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string", "minLength": 4, "maxLength": 32, "description": "工单号"},
            "reason": {"type": "string", "minLength": 4, "maxLength": 200, "description": "升级原因"},
            "priority": {"type": "string", "enum": ["普通", "紧急"], "description": "优先级，默认普通"},
        },
        "required": ["ticket_id", "reason"],
        "additionalProperties": False,
    },
    tags=("ticket", "complaint", "refund", "logistics"),
)
def escalate_to_human(
    ticket_id: str, reason: str, priority: str = "普通", _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    ticket = repo.get_ticket(ticket_id)
    if ticket is None:
        return f"没有找到工单 {ticket_id}。"
    repo.update_ticket(ticket_id, status="待人工", assignee=f"人工专员-{priority}")
    repo.add_note(ticket_id, "AI", f"升级人工（{priority}）：{reason}")
    return (
        f"工单 {ticket_id} 已升级为「待人工」（优先级：{priority}）。\n"
        f"升级原因：{reason}\n人工专员将在 24 小时内响应（紧急件 2 小时内）。"
    )


@tool(
    name="close_ticket",
    description=(
        "关闭工单并记录处理结论。当你已经解决了客户问题（或已明确告知处理结果）后调用。"
        "resolution 要写清最终结论，这是后续质检和指标统计的依据。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string", "minLength": 4, "maxLength": 32, "description": "工单号"},
            "resolution": {"type": "string", "minLength": 4, "maxLength": 300, "description": "处理结论"},
        },
        "required": ["ticket_id", "resolution"],
        "additionalProperties": False,
    },
    tags=("ticket", "order", "refund", "logistics", "invoice", "price", "coupon", "return"),
)
def close_ticket(ticket_id: str, resolution: str, _ctx: ToolContext | None = None) -> str:
    repo = _repo(_ctx)
    ticket = repo.get_ticket(ticket_id)
    if ticket is None:
        return f"没有找到工单 {ticket_id}。"
    repo.add_note(ticket_id, "AI", f"处理结论：{resolution}")
    repo.update_ticket(ticket_id, status="已关闭", closed_at=clock.stamp(), auto_closed=1)
    return f"工单 {ticket_id} 已关闭，结论已记录：{resolution}"


@tool(
    name="search_knowledge_base",
    description=(
        "在售后知识库里检索公司制度与政策条款（退换货规则、退款时效、运费规则、价保、发票、投诉赔付等）。"
        "当你要引用公司规则回答用户时使用。返回结果会带条款编号，回答时请引用它。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2, "maxLength": 120, "description": "检索关键词或问题"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5, "description": "返回条数，默认 3"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    tags=("kb", "refund", "invoice", "price", "coupon", "complaint", "logistics", "return"),
)
def search_knowledge_base(query: str, top_k: int = 3, _ctx: ToolContext | None = None) -> str:
    articles = _repo(_ctx).search_kb(query, limit=top_k)
    if not articles:
        return "知识库里没有检索到相关内容，请转人工确认，不要凭猜测回答。"
    chunks = []
    for article in articles:
        chunks.append(f"【{article['id']}】{article['title']}（{article['category']}）\n{article['content']}")
    return "\n\n".join(chunks)


# ==================================================================== 2 个关键动作（HITL）


@tool(
    name="issue_refund",
    description=(
        "为指定订单发起退款。**这是会真实动钱的关键动作，执行前必须已用 check_refund_policy 确认可退、"
        "并用 calc_refund_amount 得到金额。** 金额不得超过测算结果。"
        "系统会要求人工审批，审批通过后才会真正打款。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 4, "maxLength": 24, "description": "订单号"},
            "amount": {"type": "number", "minimum": 0.01, "maximum": 100000, "description": "退款金额（元），不得超过测算结果"},
            "reason": {"type": "string", "minLength": 4, "maxLength": 200, "description": "退款原因，需说明依据条款"},
        },
        "required": ["order_id", "amount", "reason"],
        "additionalProperties": False,
    },
    dangerous=True,
    risk_level="high",
    idempotent=True,
    tags=("refund", "order", "return"),
)
def issue_refund(
    order_id: str, amount: float, reason: str, _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    order = repo.get_order(order_id)
    if order is None:
        raise ValueError(f"订单 {order_id} 不存在")

    # 幂等：同一个 run 里对同一订单同一金额的退款只执行一次
    key = (
        _ctx.idempotency_key("refund", order_id, amount)
        if _ctx
        else f"refund:adhoc:{order_id}:{amount}"
    )
    existing = repo.find_refund(key)
    if existing:
        return (
            f"该退款已执行过（退款单号 {existing['refund_id']}，金额 {existing['amount']} 元），"
            "本次为重复请求，未重复操作。"
        )
    if order.status == "已退款":
        return f"订单 {order_id} 状态已是「已退款」，未重复操作。"

    result = repo.insert_refund(order_id, amount, reason, "AI(审批通过)", key)
    repo.update_order_status(order_id, "已退款")
    if _ctx:
        repo.log_event(_ctx.run_id, "refund", f"{order_id} 退款 {amount} 元")
    return (
        f"退款成功。退款单号 {result['refund_id']}，订单 {order_id}，金额 {amount} 元。\n"
        f"退款原因：{reason}\n"
        "到账时效：微信/支付宝 1-3 个工作日，银行卡 3-7 个工作日（KB002）。"
    )


@tool(
    name="grant_coupon",
    description=(
        "给客户发放补偿优惠券。这是有成本的关键动作，系统会要求人工审批。"
        "仅在确认是我方责任（发错/少件/延迟/质量）时使用，金额需与问题严重程度匹配，"
        "单次不超过 100 元。不要在客户仅仅表达不满时随意发券。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "minLength": 3, "maxLength": 20, "description": "客户号"},
            "amount": {"type": "number", "minimum": 1, "maximum": 100, "description": "券面额（元）"},
            "reason": {"type": "string", "minLength": 4, "maxLength": 200, "description": "发放原因，需说明我方责任"},
        },
        "required": ["customer_id", "amount", "reason"],
        "additionalProperties": False,
    },
    dangerous=True,
    risk_level="medium",
    idempotent=True,
    tags=("coupon", "complaint", "order"),
)
def grant_coupon(
    customer_id: str, amount: float, reason: str, _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    customer = repo.get_customer(customer_id)
    if customer is None:
        raise ValueError(f"客户 {customer_id} 不存在")

    key = (
        _ctx.idempotency_key("coupon", customer_id, amount, reason[:20])
        if _ctx
        else f"coupon:adhoc:{customer_id}:{amount}"
    )
    result = repo.grant_coupon(customer_id, amount, reason, key)
    if _ctx:
        repo.log_event(_ctx.run_id, "coupon", f"{customer_id} 发券 {amount} 元")
    return (
        f"已为 {customer.name}（{customer_id}）发放 {amount} 元补偿券，券码 {result['code']}。\n"
        f"发放原因：{reason}\n有效期 30 天（KB006）。"
    )


# ==================================================================== 辅助工具


@tool(
    name="add_ticket_note",
    description="给工单追加一条处理备注，用于记录中间进展。需要留痕时使用。",
    parameters={
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string", "minLength": 4, "maxLength": 32, "description": "工单号"},
            "content": {"type": "string", "minLength": 2, "maxLength": 300, "description": "备注内容"},
        },
        "required": ["ticket_id", "content"],
        "additionalProperties": False,
    },
    tags=("ticket", "order", "refund", "logistics", "invoice"),
)
def add_ticket_note(ticket_id: str, content: str, _ctx: ToolContext | None = None) -> str:
    repo = _repo(_ctx)
    if repo.get_ticket(ticket_id) is None:
        return f"没有找到工单 {ticket_id}。"
    repo.add_note(ticket_id, "AI", content)
    return f"已为工单 {ticket_id} 追加备注。"


@tool(
    name="send_notification",
    description=(
        "向客户发送通知（短信/APP 推送），用于告知处理结果。"
        "注意：本工具只负责发送，不代表业务动作已完成 —— 必须先完成实际处理再通知。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "minLength": 3, "maxLength": 20, "description": "客户号"},
            "channel": {"type": "string", "enum": ["短信", "APP推送"], "description": "通知渠道"},
            "content": {"type": "string", "minLength": 2, "maxLength": 200, "description": "通知内容"},
        },
        "required": ["customer_id", "content"],
        "additionalProperties": False,
    },
    tags=("ticket", "order", "refund", "logistics", "complaint", "coupon", "price", "invoice"),
)
def send_notification(
    customer_id: str, content: str, channel: str = "短信", _ctx: ToolContext | None = None
) -> str:
    repo = _repo(_ctx)
    customer = repo.get_customer(customer_id)
    if customer is None:
        return f"没有找到客户 {customer_id}。"
    masked = customer.phone[:3] + "****" + customer.phone[-4:]
    if _ctx:
        repo.log_event(_ctx.run_id, "notify", f"{customer_id} {channel}: {content[:60]}")
    return f"已通过{channel}向 {customer.name}（{masked}）发送通知：{content}"


@tool(
    name="query_business_db",
    description=(
        "对售后业务库执行**只读** SQL 查询（仅允许 SELECT），用于自定义统计与排查。"
        "需要按条件筛选、聚合、排序业务数据时使用。"
        "注意：这会直接查生产表，因此只允许 SELECT —— 任何写操作都会被安全策略拒绝。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "sql": {
                "type": "string", "minLength": 6, "maxLength": 600,
                "description": "SELECT 语句，例如 SELECT status, COUNT(*) FROM tickets GROUP BY status",
            }
        },
        "required": ["sql"],
        "additionalProperties": False,
    },
    dangerous=True,               # 交 Guardrail 走 readonly_sql 内容级策略
    risk_level="medium",
    policy="readonly_sql",        # ← 内容级硬约束：非 SELECT 一律拒绝，无需人工确认
    tags=("analysis",),
)
def query_business_db(sql: str, _ctx: ToolContext | None = None) -> str:
    repo = _repo(_ctx)
    rows = repo.query(sql)
    if not rows:
        return "查询成功，但没有结果。"
    columns = list(rows[0].keys())
    lines = [" | ".join(columns)]
    lines += [" | ".join(str(r.get(c)) for c in columns) for r in rows[:30]]
    if len(rows) > 30:
        lines.append(f"…（共 {len(rows)} 行，仅显示前 30 行）")
    return "\n".join(lines)


@tool(
    name="get_current_time",
    description="获取当前业务时间。涉及「今天」「几天内」「过期了吗」这类相对时间判断时使用。",
    parameters={"type": "object", "properties": {}},
    tags=("order", "refund", "price", "logistics", "ticket", "kb", "invoice", "return"),
)
def get_current_time(_ctx: ToolContext | None = None) -> str:
    return f"当前时间：{clock.stamp()}"


@tool(
    name="calculator",
    description="计算数学表达式，支持 + - * / % ** 和括号。需要精确计算时使用。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "minLength": 1, "maxLength": 200, "description": "算式，例如 (99*2+12)*0.9"}
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    tags=("order", "refund", "price", "coupon"),
)
def calculator(expression: str) -> str:
    import ast
    import operator

    bin_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
            return bin_ops[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](evaluate(node.operand))
        raise ValueError("只支持数字和 + - * / % ** 运算")

    try:
        result = evaluate(ast.parse(expression, mode="eval"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"算式无法计算：{exc}") from exc
    return f"{expression} = {result}"
