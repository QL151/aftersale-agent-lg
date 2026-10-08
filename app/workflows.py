"""8 条确定性工作流：高频意图不经过模型，直接按固定步骤执行。

为什么这些任务不该交给大模型？
  看一条真实工单：「帮我查下物流，三天没动了」
  它的处理路径**永远是**：
      取订单 → 查轨迹 → 判断是否超过 72 小时无更新（KB008）→ 按规则回复/建单
  让模型自由发挥的代价是：贵（每轮塞 19 个工具描述）、慢（多轮往返）、
  而且**可能忘了查政策就直接承诺赔偿**。

  所以这 8 条流程用代码写死，收益是：
    · 成本：0 token（不调模型）
    · 稳定性：同样的输入必然得到同样的处理路径，可回归测试
    · 可解释：每一步都有明确的业务依据（引用条款号）
    · 快：一次工具往返，不是 4 轮 ReAct

设计要点：**所有工作流共用同一套节点原语**（resolve / policy / respond / hitl / finalize），
所以新增一条工作流通常只有 5~10 行，而不是复制粘贴一整段。
"""

from __future__ import annotations

from typing import Any

from . import clock
from .graph.core import END, Command, StateGraph
from .hitl import ApprovalManager
from .schemas import AgentState

# ==================================================================== 工作流工具箱


class WorkflowKit:
    """节点共享的依赖与动作。工作流只描述"业务流程"，不碰底层细节。"""

    def __init__(self, executor: Any, repo: Any, tracer: Any = None,
                 hitl: ApprovalManager | None = None, settings: Any = None) -> None:
        self.ex = executor
        self.repo = repo
        self.tracer = tracer
        self.hitl = hitl or ApprovalManager(repo)
        from .config import settings as global_settings

        self.s = settings or global_settings

    # ---------------------------------------------------------- 基础
    def step(self, state: AgentState, kind: str, detail: dict) -> None:
        self.ex._record_step(state, kind, detail)

    def reply(self, state: AgentState, text: str) -> dict:
        return {"answer": text}

    def ask(self, state: AgentState, question: str) -> dict:
        return {"needs_clarify": True, "clarify_question": question, "answer": question}

    def tool(self, state: AgentState, name: str, arguments: dict) -> str:
        """执行工具并返回**面向用户的结果文本**。

        刻意返回 result.output 而不是 to_llm_content()：
        确定性工作流的答复是直接给客户看的，不能把 "[执行成功] xxx 返回：" 这种
        写给模型的协议包装漏出去。
        """
        result = self.ex.call(state, name, arguments)
        if result.ok:
            return result.output
        if result.blocked:
            return f"（该操作被安全策略拒绝：{result.block_reason}）"
        return f"（操作未成功：{result.error}）"

    # ---------------------------------------------------------- 订单解析
    def resolve_order(self, state: AgentState) -> dict:
        """从槽位解析出唯一订单。解析不到就追问 —— **绝不猜**。"""
        slots = state.slots or {}
        order_id = slots.get("order_id") or ""
        name = slots.get("customer_name") or ""
        customer_id = slots.get("customer_id") or ""

        if not order_id and customer_id:
            customer = self.repo.get_customer(customer_id)
            if customer:
                name = customer.name

        if order_id:
            text = self.tool(state, "get_order_detail", {"order_id": order_id})
            order = self.repo.get_order(order_id)
            if order is None:
                return self.ask(state, f"没有查到订单 {order_id}，麻烦确认一下订单号是否正确？")
            state.put("order", order.public())
            state.put("customer_id", order.customer_id)
            return {}

        if name:
            text = self.tool(state, "search_orders_by_customer", {"customer_name": name, "limit": 5})
            customers = self.repo.find_customers_by_name(name)
            if not customers:
                return self.ask(state, f"没有找到客户「{name}」的账号，方便提供订单号吗？")
            orders = self.repo.list_orders_by_customer(customers[0].customer_id, limit=5)
            if not orders:
                return self.ask(state, f"客户「{name}」名下暂时没有订单，请确认一下姓名或提供订单号。")
            if len(orders) > 1:
                options = "、".join(f"{o.order_id}（{o.sku_name}）" for o in orders[:3])
                return self.ask(state, f"查到多个订单：{options}。请问是哪一个？")
            order = orders[0]
            state.put("order", order.public())
            state.put("customer_id", customers[0].customer_id)
            state.slots["order_id"] = order.order_id
            return {}

        return self.ask(
            state, "为了帮你处理，请提供订单号（例如 SO2026000249），或告诉我下单时用的姓名。"
        )

    # ---------------------------------------------------------- 工单
    def ensure_ticket(self, state: AgentState) -> str:
        """每个会话都在工单系统里留痕 —— 这是"全自动闭环率"能被统计的前提。"""
        existing = state.fact("ticket_id")
        if existing:
            return existing
        order = state.fact("order") or {}
        order_id = order.get("order_id") or state.slots.get("order_id") or ""
        customer_id = state.fact("customer_id") or state.slots.get("customer_id") or ""
        if not order_id:
            return ""
        text = self.tool(state, "create_ticket", {
            "order_id": order_id,
            "intent": state.intent or "售后咨询",
            "title": f"{state.intent or '售后'}：{state.task[:40]}",
        })
        ticket_id = ""
        for line in text.splitlines():
            if line.startswith("已创建工单"):
                # 返回值形如「已创建工单 T261002C6D04D（状态：处理中，处理人：AI）。」
                # 原写法 line.split()[1].strip("（") 只能剥掉**首尾**的括号字符，
                # 而括号在中间，于是 ticket_id 被拼上「（状态：…）。」整段后缀，
                # 后续 close_ticket 就报「没有找到工单 Txxx（状态：…）」。
                # 这里按左括号切一刀，只取编号本身。
                ticket_id = line.split()[1].split("（")[0].strip()
                break
        if ticket_id:
            state.put("ticket_id", ticket_id)
        return ticket_id

    def finish(self, state: AgentState, resolution: str, handover: bool = False) -> dict:
        """收尾：关单 / 记结论 / 沉淀长期偏好。"""
        ticket_id = state.fact("ticket_id") or ""
        if ticket_id and not handover:
            self.tool(state, "close_ticket", {"ticket_id": ticket_id, "resolution": resolution})
        customer_id = state.fact("customer_id") or state.slots.get("customer_id") or ""
        if customer_id and self.ex.profile is not None:
            try:
                self.ex.profile.learn_from_run(
                    customer_id, state.intent or "unknown",
                    {
                        "resolution": resolution[:40],
                        "needs_manual": handover,
                        "summary": f"{state.task[:30]} → {resolution[:40]}",
                        "tags": ["需人工复核"] if handover else [],
                    },
                    run_id=state.run_id,
                )
            except Exception:  # noqa: BLE001
                pass
        auto = (not handover) and (not state.approval_denied)
        return {"auto_closed": auto, "handover": handover}

    def escalate(self, state: AgentState, reason: str, priority: str = "普通") -> dict:
        ticket_id = self.ensure_ticket(state)
        if ticket_id:
            self.tool(state, "escalate_to_human",
                      {"ticket_id": ticket_id, "reason": reason, "priority": priority})
        return self.finish(state, f"已转人工：{reason}", handover=True)


# ==================================================================== 通用节点


def node_resolve(kit: WorkflowKit):
    """所有工作流共用的入口节点：解析订单 → 建单留痕。解析不到就由条件边转追问。"""

    def action(state: AgentState):
        update = kit.resolve_order(state)
        if update.get("needs_clarify"):
            return update
        kit.ensure_ticket(state)
        return {}

    return action


def make_after_resolve(target: str):
    """条件边工厂：解析成功去业务流程首节点，失败去追问节点。

    target 每条工作流不同（gather / policy / act / check），所以做成工厂，
    避免 8 条工作流各写一个几乎一样的判断函数。
    """

    def route(state: AgentState) -> str:
        return "clarify" if state.needs_clarify else target

    return route


def node_resolve_lenient(kit: WorkflowKit):
    """宽松版解析：拿不到订单也继续走流程。

    给"优惠券为什么用不了"这类**不以订单为中心**的咨询用：
    强行要求订单号只会让客户多答一轮，而这些问题的答案本来就和订单无关。
    """

    def action(state: AgentState):
        slots = state.slots or {}
        customer_id = slots.get("customer_id") or ""
        if not customer_id and slots.get("customer_name"):
            found = kit.repo.find_customers_by_name(slots["customer_name"])
            if len(found) == 1:
                customer_id = found[0].customer_id
        if customer_id:
            state.put("customer_id", customer_id)
            state.slots["customer_id"] = customer_id

        # 顺手解析订单（有就用，没有不追问）
        if slots.get("order_id"):
            saved = state.needs_clarify
            kit.resolve_order(state)
            state.needs_clarify = saved
            state.clarify_question = ""
        if state.fact("customer_id"):
            kit.ensure_ticket(state)
        return {}

    return action


def make_after_lenient(target: str):
    def route(state: AgentState) -> str:
        return target

    return route


def node_clarify(kit: WorkflowKit):
    def action(state: AgentState):
        return {"answer": state.clarify_question or "需要更多信息才能继续。", "route": state.route}

    return action


# ==================================================================== 8 条工作流


def build_logistics_query(kit: WorkflowKit) -> StateGraph:
    """物流查询：取订单 → 查轨迹 → 按 KB008 判定异常 → 回复（异常则建单跟进）。"""
    graph = StateGraph("logistics_query")

    def gather(state: AgentState):
        order = state.fact("order") or {}
        text = kit.tool(state, "get_logistics_trace", {"order_id": order.get("order_id", "")})
        state.put("logistics", text)
        abnormal = "异常" in text or "停滞" in text
        state.put("abnormal", abnormal)
        return {}

    def respond(state: AgentState):
        order = state.fact("order") or {}
        text = state.fact("logistics") or ""
        if state.fact("abnormal"):
            body = (
                f"订单 {order.get('order_id')} 的物流出现异常，我已为你登记跟进。\n\n"
                f"{text}\n\n"
                "处理承诺：已向承运商发起查询，48 小时内未恢复将按丢件处理，"
                "可选择补发或全额退款（依据 KB008 物流异常处理流程）。"
            )
            return {"answer": body, "handover": False}
        return {
            "answer": f"订单 {order.get('order_id')} 的物流状态如下：\n\n{text}\n\n"
                      "如果物流超过 72 小时没有更新，可以告诉我，我帮你走异常件流程。"
        }

    graph.add_node("resolve", node_resolve(kit))
    graph.add_node("clarify", node_clarify(kit))
    graph.add_node("gather", gather)
    graph.add_node("respond", respond)
    graph.add_node("finalize", lambda s: kit.finish(s, "物流查询已答复"))
    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", make_after_resolve("gather"))
    graph.add_edge("clarify", END)
    graph.add_edge("gather", "respond")
    graph.add_edge("respond", "finalize")
    graph.add_edge("finalize", END)
    return graph


def build_refund_progress(kit: WorkflowKit) -> StateGraph:
    """退款进度查询：查订单状态 + 退款记录 → 按 KB002 给到账时效。"""

    def gather(state: AgentState):
        order = state.fact("order") or {}
        order_id = order.get("order_id", "")
        status = order.get("status", "")
        refund = None
        rows = kit.repo.query(
            "SELECT refund_id, amount, created_at FROM refunds WHERE order_id = ? ORDER BY created_at DESC LIMIT 1",
            (order_id,),
        )
        if rows:
            refund = rows[0]
        state.put("refund_record", refund)
        state.put("order_status", status)
        return {}

    def respond(state: AgentState):
        order = state.fact("order") or {}
        refund = state.fact("refund_record")
        if refund:
            days = clock.days_since(refund["created_at"]) or 0
            body = (
                f"订单 {order.get('order_id')} 的退款已于 {refund['created_at']} 发起，"
                f"金额 {refund['amount']} 元，退款单号 {refund['refund_id']}，已处理 {days:.1f} 天。\n"
                "到账时效（KB002）：微信/支付宝 1-3 个工作日，银行卡 3-7 个工作日。\n"
            )
            if days > 7:
                body += (
                    "该笔退款已超过 7 个工作日，属于异常情况。我这就为你发起支付渠道查询，"
                    "并升级人工专员跟进。"
                )
                return {"answer": body, "handover": True}
            return {"answer": body + "你的退款正在正常处理中，请留意到账。"}
        return {
            "answer": (
                f"订单 {order.get('order_id')} 当前状态为「{order.get('status')}」，"
                "系统里没有查到退款记录。\n"
                "如果你已经申请过退款，请提供申请时间，我帮你核实；"
                "如果还没申请，可以告诉我，我帮你走退款流程。"
            )
        }

    graph = StateGraph("refund_progress")
    graph.add_node("resolve", node_resolve(kit))
    graph.add_node("clarify", node_clarify(kit))
    graph.add_node("gather", gather)
    graph.add_node("respond", respond)
    graph.add_node("finalize", lambda s: kit.finish(
        s, "退款进度已答复" + ("（超期，已转人工）" if s.handover else ""), handover=s.handover))
    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", make_after_resolve("gather"))
    graph.add_edge("clarify", END)
    graph.add_edge("gather", "respond")
    graph.add_edge("respond", "finalize")
    graph.add_edge("finalize", END)
    return graph


def build_refund_request(kit: WorkflowKit) -> StateGraph:
    """退款申请（最关键的一条）：查订单 → 政策校验 → 金额测算 → 风控判断
    → [HITL 人工审批] → 执行退款 → 答复。

    这是"人机协同"的主战场：动钱的动作必须有人点头，
    而且审批是**图的节点**，所以审批期间进程可以重启。
    """

    def policy(state: AgentState):
        order = state.fact("order") or {}
        quality = bool(state.slots.get("quality_issue"))
        text = kit.tool(state, "check_refund_policy",
                        {"order_id": order.get("order_id", ""), "quality_issue": quality})
        state.put("policy_text", text)
        state.put("policy_allowed", "是否可退：可以" in text)
        state.put("policy_needs_manual", "需要人工复核：是" in text)
        return {}

    def quote(state: AgentState):
        order = state.fact("order") or {}
        quality = bool(state.slots.get("quality_issue"))
        kit.tool(state, "calc_refund_amount",
                 {"order_id": order.get("order_id", ""), "quality_issue": quality})
        from .db import calc_refund, check_policy

        full = kit.repo.get_order(order.get("order_id", ""))
        rule = kit.repo.get_rule(full.category) if full else None
        if full is None:
            return {"answer": "订单数据异常，请稍后重试或转人工。"}
        quotation = calc_refund(full, kit.repo.promo_deduct(full.order_id), rule,
                                quality_issue=quality)
        state.put("refund_quote", {
            "order_id": quotation.order_id, "total": quotation.total,
            "breakdown": quotation.breakdown,
        })
        return {}

    def decide(state: AgentState):
        """风控判断：谁必须走人工，谁可以自动放行。"""
        if not state.fact("policy_allowed"):
            return Command(goto="deny")
        quote_data = state.fact("refund_quote") or {}
        amount = float(quote_data.get("total", 0))
        customer_id = state.fact("customer_id") or ""
        customer = kit.repo.get_customer(customer_id) if customer_id else None
        risk = customer.risk_score if customer else 0.0

        # 提示注入检测：只要用户话里出现"不用审核""直接全额退""把金额改成"这类话术，
        # 无论金额多小都必须人工过一眼 —— 攻击者最想要的就是"绕过阈值自动放行"。
        from .guardrail import Guardrail

        injection = Guardrail.detect_injection(state)

        needs_human = (
            amount > kit.s.approval_auto_approve_max
            or risk >= 0.5
            or bool(state.fact("policy_needs_manual"))
            or bool(injection)
        )
        state.put("needs_human", needs_human)
        state.put("risk_score", risk)
        if needs_human:
            reasons = []
            if amount > kit.s.approval_auto_approve_max:
                reasons.append(f"金额 {amount} 元 > 自动放行上限 {kit.s.approval_auto_approve_max} 元")
            if risk >= 0.5:
                reasons.append(f"客户风险评分 {risk} ≥ 0.5")
            if state.fact("policy_needs_manual"):
                reasons.append("政策判定需要人工复核")
            if injection:
                reasons.append(f"检测到疑似提示注入「{injection}」，强制人工复核")
            state.put("approval_reason", "；".join(reasons))
            return Command(goto="approval")
        return Command(goto="execute")

    def approval(state: AgentState):
        """HITL 审批节点 —— 唯一会 interrupt() 的地方，所以 resume 只重跑这一个节点。"""
        order = state.fact("order") or {}
        quote_data = state.fact("refund_quote") or {}
        amount = float(quote_data.get("total", 0))
        update = kit.hitl.gate(
            state,
            action="issue_refund",
            arguments={"order_id": order.get("order_id"), "amount": amount,
                       "reason": "客户申请退款"},
            reason=state.fact("approval_reason", "退款需人工审批"),
            risk_level="high",
            amount=amount,
            force=True,                # 工作流已在 decide 节点判过风控，这里不再做小额免打扰
        )
        decision = update.get("facts", {}).get("_approval", {})
        if kit.tracer is not None:
            kit.tracer.approval(
                decision.get("approval_id", ""), "issue_refund",
                "approved" if decision.get("approved") else "rejected", amount,
                decision.get("note", ""),
            )
        # 必须把 gate 的状态更新带出去（Command.update）：
        # 否则 approvals / approval_denied / _approval 这些字段会被丢掉，
        # 结果就是"审批确实发生过，但指标和后续判断都看不到"。
        if not decision.get("approved"):
            return Command(goto="denied", update=update)
        return Command(goto="execute", update=update)

    def execute(state: AgentState):
        order = state.fact("order") or {}
        quote_data = state.fact("refund_quote") or {}
        amount = float(quote_data.get("total", 0))
        result = kit.ex.call(
            state, "issue_refund",
            {"order_id": order.get("order_id"), "amount": amount,
             "reason": "客户申请退款，政策校验通过并经人工审批"},
            approved=True,
        )
        state.put("refund_result", result.output if result.ok else f"（退款未成功：{result.error}）")
        return {}

    def respond(state: AgentState):
        order = state.fact("order") or {}
        quote_data = state.fact("refund_quote") or {}
        breakdown = "\n".join(f"  · {b}" for b in quote_data.get("breakdown", []))
        result = state.fact("refund_result") or ""
        return {
            "answer": (
                f"订单 {order.get('order_id')} 的退款已处理完成。\n\n"
                f"退款明细：\n{breakdown}\n\n{result}"
            )
        }

    def deny(state: AgentState):
        return {
            "answer": (
                f"很抱歉，订单 {(state.fact('order') or {}).get('order_id')} 暂时不符合退款条件。\n\n"
                f"{state.fact('policy_text')}\n\n"
                "如果你认为属于质量问题或有不属于规则覆盖的特殊情况，"
                "我可以帮你转人工专员复核。"
            )
        }

    def denied(state: AgentState):
        decision = (state.fact("_approval") or {})
        return Command(goto="escalate")

    graph = StateGraph("refund_request")
    graph.add_node("resolve", node_resolve(kit))
    graph.add_node("clarify", node_clarify(kit))
    graph.add_node("policy", policy)
    graph.add_node("quote", quote)
    graph.add_node("decide", decide)
    graph.add_node("approval", approval)
    graph.add_node("execute", execute)
    graph.add_node("respond", respond)
    graph.add_node("deny", deny)
    graph.add_node("denied", denied)
    graph.add_node("escalate", lambda s: kit.escalate(
        s, "退款申请被人工驳回，需专员复核客户诉求", priority="紧急"))
    graph.add_node("finalize", lambda s: kit.finish(
        s, "退款已完成" if s.fact("refund_result") else "不符合退款条件，已解释条款",
        handover=s.handover))

    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", make_after_resolve("policy"))
    graph.add_edge("clarify", END)
    graph.add_edge("policy", "quote")
    graph.add_edge("quote", "decide")
    graph.add_edge("approval", "execute")      # 实际跳转由 Command 决定，这里给编译器一个出边
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", "finalize")
    graph.add_edge("deny", "finalize")
    graph.add_edge("denied", "escalate")
    graph.add_edge("escalate", END)
    graph.add_edge("finalize", END)
    return graph


def build_return_exchange(kit: WorkflowKit) -> StateGraph:
    """退货换货：政策校验 → 换货需同款有货且 7 天内 → 给寄回指引。"""

    def policy(state: AgentState):
        order = state.fact("order") or {}
        kit.tool(state, "check_refund_policy", {"order_id": order.get("order_id", "")})
        kit.tool(state, "search_knowledge_base", {"query": "退货换货 操作指引"})
        from .db import check_policy

        full = kit.repo.get_order(order.get("order_id", ""))
        rule = kit.repo.get_rule(full.category) if full else None
        verdict = check_policy(full, rule) if full else None
        state.put("return_verdict", {
            "allowed": verdict.allowed, "reason": verdict.reason,
            "clause": verdict.clause, "needs_manual": verdict.needs_manual,
        } if verdict else {"allowed": False, "reason": "订单数据缺失", "needs_manual": True})
        state.put("category", full.category if full else "")
        return {}

    def respond(state: AgentState):
        order = state.fact("order") or {}
        verdict = state.fact("return_verdict") or {}
        if not verdict.get("allowed"):
            return {
                "answer": (
                    f"订单 {order.get('order_id')} 暂不支持退货/换货。\n"
                    f"依据：{verdict.get('clause', '')} —— {verdict.get('reason', '')}\n"
                    "如果是质量问题，请提供照片凭证，我可以帮你走质量问题通道。"
                ),
                "handover": bool(verdict.get("needs_manual")),
            }
        return {
            "answer": (
                f"订单 {order.get('order_id')}（{state.fact('category')}）符合退货换货条件。\n"
                f"依据：{verdict.get('clause', '')} —— {verdict.get('reason', '')}\n\n"
                "操作指引（KB011）：\n"
                "  1. 在「我的订单」提交退换申请，选择退货或换货（换货需同款同规格有库存）；\n"
                "  2. 按页面提示的地址寄回，保留运费凭证；\n"
                "  3. 商家签收后 48 小时内完成退款或发出换货商品；\n"
                "  4. 非质量问题的退货运费由买家承担（KB003）。\n"
                "需要我直接帮你提交退换申请吗？"
            ),
            "auto_closed_hint": True,
        }

    graph = StateGraph("return_exchange")
    graph.add_node("resolve", node_resolve(kit))
    graph.add_node("clarify", node_clarify(kit))
    graph.add_node("policy", policy)
    graph.add_node("respond", respond)
    graph.add_node("finalize", lambda s: kit.finish(
        s, "退换货指引已答复" + ("（需人工复核）" if s.handover else ""), handover=s.handover))
    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", make_after_resolve("policy"))
    graph.add_edge("clarify", END)
    graph.add_edge("policy", "respond")
    graph.add_edge("respond", "finalize")
    graph.add_edge("finalize", END)
    return graph


def build_invoice_issue(kit: WorkflowKit) -> StateGraph:
    """发票问题：查询状态 → 按动作申请/换开/催办。"""

    def act(state: AgentState):
        order = state.fact("order") or {}
        task = state.task
        if any(k in task for k in ("换开", "重开", "改抬头", "抬头错了")):
            action = "换开"
        elif any(k in task for k in ("没开", "开票", "要发票", "需要发票", "开发票", "报销")):
            action = "申请开票"
        elif any(k in task for k in ("催", "怎么还没", "什么时候")):
            action = "催办"
        else:
            action = "查询"
        title_type = "企业" if any(k in task for k in ("公司", "企业", "税号", "专票")) else "个人"
        text = kit.tool(state, "handle_invoice",
                        {"order_id": order.get("order_id", ""), "action": action,
                         "title_type": title_type})
        state.put("invoice_action", action)
        state.put("invoice_text", text)
        return {}

    def respond(state: AgentState):
        order = state.fact("order") or {}
        action = state.fact("invoice_action", "查询")
        return {
            "answer": (
                f"订单 {order.get('order_id')} 的发票处理结果（{action}）：\n\n"
                f"{state.fact('invoice_text')}\n\n"
                "如还有其他发票问题（例如改抬头、补开企业专票），告诉我即可。"
            )
        }

    graph = StateGraph("invoice_issue")
    graph.add_node("resolve", node_resolve(kit))
    graph.add_node("clarify", node_clarify(kit))
    graph.add_node("act", act)
    graph.add_node("respond", respond)
    graph.add_node("finalize", lambda s: kit.finish(s, "发票问题已处理"))
    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", make_after_resolve("act"))
    graph.add_edge("clarify", END)
    graph.add_edge("act", "respond")
    graph.add_edge("respond", "finalize")
    graph.add_edge("finalize", END)
    return graph


def build_price_protection(kit: WorkflowKit) -> StateGraph:
    """价保补差：需要在对话里拿到现价；拿不到就追问。"""

    def check(state: AgentState):
        order = state.fact("order") or {}
        current = state.slots.get("current_price")
        if current is None:
            return Command(goto="ask_price")
        text = kit.tool(state, "check_price_protection", {
            "order_id": order.get("order_id", ""),
            "current_price": float(current),
            "is_promo": bool(state.slots.get("is_promo")),
        })
        state.put("price_text", text)
        eligible = "满足价保条件" in text
        state.put("price_eligible", eligible)
        if eligible:
            import re

            match = re.search(r"可补差价\s*([\d.]+)\s*元", text)
            state.put("price_diff", float(match.group(1)) if match else 0.0)
        return Command(goto="respond")

    def ask_price(state: AgentState):
        order = state.fact("order") or {}
        return kit.ask(
            state,
            f"订单 {order.get('order_id')} 可以走价保。请告诉我现在的售价是多少元？"
            "（如果是秒杀/拼团价，请说明一下，这类价差不参与价保）",
        )

    def respond(state: AgentState):
        order = state.fact("order") or {}
        text = state.fact("price_text") or ""
        if state.fact("price_eligible"):
            diff = float(state.fact("price_diff") or 0)
            return {
                "answer": (
                    f"{text}\n\n"
                    f"已为你登记价保补差 {diff} 元，将以原路退回或余额形式在 3 个工作日内到账。"
                    "（依据 KB005，每个订单仅可申请一次价保）"
                )
            }
        return {"answer": f"{text}\n\n如需进一步核实，我可以帮你转人工专员。"}

    graph = StateGraph("price_protection")
    graph.add_node("resolve", node_resolve(kit))
    graph.add_node("clarify", node_clarify(kit))
    graph.add_node("check", check)
    graph.add_node("ask_price", ask_price)
    graph.add_node("respond", respond)
    graph.add_node("finalize", lambda s: kit.finish(
        s, "价保已答复" if not s.needs_clarify else "价保待补充现价"))
    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", make_after_resolve("check"))
    graph.add_edge("clarify", END)
    graph.add_edge("ask_price", END)
    graph.add_edge("check", "respond")
    graph.add_edge("respond", "finalize")
    graph.add_edge("finalize", END)
    return graph


def build_coupon_promo(kit: WorkflowKit) -> StateGraph:
    """优惠券咨询：查客户已持有券 + 按 KB006 解释规则；符合条件可申请补偿券（HITL）。"""

    def gather(state: AgentState):
        customer_id = state.fact("customer_id") or state.slots.get("customer_id") or ""
        coupons = kit.repo.list_coupons(customer_id) if customer_id else []
        state.put("coupons", coupons)
        kit.tool(state, "search_knowledge_base", {"query": "优惠券 使用与补偿规则"})
        return {}

    def respond(state: AgentState):
        coupons = state.fact("coupons") or []
        if coupons:
            lines = "\n".join(
                f"  · {c['code']}  {c['amount']} 元  {c['status']}  有效期至 {c['expire_at'][:10]}"
                for c in coupons[:5]
            )
            return {
                "answer": (
                    f"你账户下有 {len(coupons)} 张优惠券：\n{lines}\n\n"
                    "使用规则（KB006）：优惠券不可提现、不可转赠，逾期作废；"
                    "单笔订单限用一张，具体以结算页展示为准。\n"
                    "如果是因平台或商家原因导致订单取消，可以申请等额补偿券 —— 告诉我订单号即可。"
                )
            }
        return {
            "answer": (
                "你账户下目前没有可用的优惠券。\n"
                "常见原因（KB006）：已过期、已在下单时使用、或该活动券有品类限制。\n"
                "如果是因平台或商家原因导致订单取消，可以申请等额补偿券，告诉我订单号我帮你核实。"
            )
        }

    graph = StateGraph("coupon_promo")
    graph.add_node("resolve", node_resolve_lenient(kit))
    graph.add_node("clarify", node_clarify(kit))
    graph.add_node("gather", gather)
    graph.add_node("respond", respond)
    graph.add_node("finalize", lambda s: kit.finish(s, "优惠券咨询已答复"))
    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", make_after_lenient("gather"))
    graph.add_edge("clarify", END)
    graph.add_edge("gather", "respond")
    graph.add_edge("respond", "finalize")
    graph.add_edge("finalize", END)
    return graph


def build_complaint_escalation(kit: WorkflowKit) -> StateGraph:
    """投诉升级：先止损（查清事实、给补偿方案）→ 再转人工专员（客户明确要人工必须转）。"""

    def gather(state: AgentState):
        order = state.fact("order") or {}
        order_id = order.get("order_id", "")
        customer_id = state.fact("customer_id") or ""
        if order_id:
            kit.tool(state, "get_logistics_trace", {"order_id": order_id})
        if customer_id:
            kit.tool(state, "get_customer_profile", {"customer_id": customer_id})
        kit.tool(state, "search_knowledge_base", {"query": "投诉升级与赔付标准"})
        customer = kit.repo.get_customer(customer_id) if customer_id else None
        state.put("customer_risk", customer.risk_score if customer else 0.0)
        state.put("customer_level", customer.level if customer else "普通")
        return {}

    def respond(state: AgentState):
        order = state.fact("order") or {}
        priority = "紧急" if any(
            k in state.task for k in ("投诉", "曝光", "315", "消协", "差评")
        ) else "普通"
        reason = (
            f"客户投诉（{state.fact('customer_level', '普通')}会员，"
            f"风险评分 {state.fact('customer_risk', 0)}）：{state.task[:80]}"
        )
        ticket_id = state.fact("ticket_id") or ""
        if ticket_id:
            kit.tool(state, "escalate_to_human",
                     {"ticket_id": ticket_id, "reason": reason, "priority": priority})
        order_note = f"订单 {order.get('order_id')}，" if order.get("order_id") else ""
        return {
            "answer": (
                "非常抱歉给你带来不好的体验，你的问题我已登记并升级处理。\n\n"
                f"{order_note}已转人工专员（优先级：{priority}），"
                "专员会在 2 小时内（紧急件）联系你。\n\n"
                "赔付标准（KB007）：少件按缺失商品实付金额赔付；虚假发货赔付订单实付金额的 30%。\n"
                "专员会核对你的具体情况后给出处理方案。"
            )
        }

    graph = StateGraph("complaint_escalation")
    graph.add_node("resolve", node_resolve(kit))
    graph.add_node("clarify", node_clarify(kit))
    graph.add_node("gather", gather)
    graph.add_node("respond", respond)
    graph.add_node("finalize", lambda s: kit.finish(s, "投诉已升级人工专员", handover=True))
    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", make_after_resolve("gather"))
    graph.add_edge("clarify", END)
    graph.add_edge("gather", "respond")
    graph.add_edge("respond", "finalize")
    graph.add_edge("finalize", END)
    return graph


# ==================================================================== 注册表

BUILDERS = {
    "logistics_query": build_logistics_query,
    "refund_progress": build_refund_progress,
    "refund_request": build_refund_request,
    "return_exchange": build_return_exchange,
    "invoice_issue": build_invoice_issue,
    "price_protection": build_price_protection,
    "coupon_promo": build_coupon_promo,
    "complaint_escalation": build_complaint_escalation,
}


def build_workflow(name: str, kit: WorkflowKit) -> StateGraph | None:
    builder = BUILDERS.get(name)
    return builder(kit) if builder else None


def workflow_names() -> list[str]:
    return list(BUILDERS)
