"""长尾自主 Agent：ReAct + Function Calling，用同一套状态机引擎跑。

什么时候走这里？
  意图路由层判定"没有确定性工作流覆盖"时（例如"我想把两个订单的发票合并开一张"、
  "帮我分析一下最近退款率最高的品类"）。这类需求长尾、组合多，写工作流不划算。

它和 8 条工作流共用：
  · 同一套工具（19 个）、同一套护栏、同一套 Trace、同一套 HITL
  · 同一个状态机引擎 —— 所以循环上限、失败回退、断点续跑都自动生效

与工作流最大的不同：**"下一步做什么"由模型决定**，所以必须有更硬的约束：
  · 循环上限（recursion_limit）—— 防止模型反复试同一个工具
  · 重复调用检测（Guardrail）—— 同工具同参数最多 N 次
  · 关键动作必须审批 —— 且审批**在独立节点里**，
    否则 resume 会把"模型调用"整段重跑（重复计费 + 可能换一个工具调用，这就是经典的
    "重放不确定副作用"问题）。
"""

from __future__ import annotations

import json
from typing import Any

from .graph.core import END, Command, StateGraph
from .schemas import AgentState

SYSTEM_PROMPT = """你是电商平台的售后客服 Agent，负责处理客户的售后问题。

工作原则：
1. **先查事实，再下结论**。订单、物流、金额、政策条款一律用工具查，绝不凭记忆编造。
   尤其：任何"能不能退""退多少"的结论，必须先调用 check_refund_policy / calc_refund_amount。
2. 一次只调用一个工具，拿到结果再决定下一步。
3. 引用公司规则时，要带上条款编号（例如「依据 KB001 七天无理由退货规则」）。
4. 退款、发券这类关键动作会触发人工审批。提交后如果被驳回，不要反复提交，
   改为向客户说明，并转人工专员。
5. 工具被拦截或报错时，不要重复同样的调用：换参数、换工具，或者如实说明做不到。
6. 信息不足时**先问清楚**（订单号、现价、是否拆封、是否质量问题），不要猜。
7. 无法闭环的问题（需要人工、超出权限、客户情绪激烈）就建单并转人工，不要硬撑。

输出要求：中文，简洁、有同理心、给出明确的下一步。不要暴露内部工具名和实现细节。"""


class AgentGraphBuilder:
    def __init__(self, llm: Any, executor: Any, kit: Any, tools_spec: Any,
                 cost: Any = None, tracer: Any = None, settings: Any = None) -> None:
        self.llm = llm
        self.ex = executor
        self.kit = kit
        self.tools_spec = tools_spec
        self.cost = cost
        self.tracer = tracer
        from .config import settings as global_settings

        self.s = settings or global_settings

    # ------------------------------------------------------------ 消息组装
    def _messages(self, state: AgentState) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 会话上下文（Redis / 本地记忆）：让"上一句说了什么"不丢
        memory = getattr(self.ex, "session_memory", None)
        if memory is not None and state.session_id:
            try:
                session = memory.load(state.session_id)
                messages.extend(session.context_messages(limit=6))
                visible = {k: v for k, v in (session.slots or {}).items()
                           if not k.startswith("_")}
                if visible:
                    messages.append({
                        "role": "system",
                        "content": "【本会话已确认的槽位】"
                                   + json.dumps(visible, ensure_ascii=False),
                    })
            except Exception:  # noqa: BLE001
                pass

        # 长期偏好（PostgreSQL / SQLite）
        customer_id = state.fact("customer_id") or state.slots.get("customer_id")
        profile = getattr(self.ex, "profile", None)
        if profile is not None and customer_id:
            try:
                prefs = profile.all_preferences(customer_id)
                facts = profile.recent_facts(customer_id, limit=3)
                if prefs or facts:
                    messages.append({
                        "role": "system",
                        "content": "【该客户的历史偏好与记录】"
                                   + json.dumps(prefs, ensure_ascii=False)
                                   + ("｜近期记录：" + "；".join(facts) if facts else ""),
                    })
            except Exception:  # noqa: BLE001
                pass

        if state.messages:
            messages.extend(state.messages)
        else:
            messages.append({"role": "user", "content": state.task})
        return messages

    # ------------------------------------------------------------ 节点
    def node_agent(self, state: AgentState):
        messages = self._messages(state)
        model = self.cost.pick_model(self.s.llm_model) if self.cost else self.s.llm_model
        reply = self.llm.chat(messages, tools=self.tools_spec, model=model)

        state.prompt_tokens += reply.prompt_tokens
        state.completion_tokens += reply.completion_tokens
        state.model_used = reply.model or model

        if self.cost is not None:
            outcome = self.cost.charge(reply.model or model, reply.prompt_tokens,
                                       reply.completion_tokens)
            state.cost_yuan = self.cost.state.run_cost
            state.downgraded = self.cost.state.downgraded
            if self.tracer is not None:
                self.tracer.cost(self.cost.state.run_tokens, self.cost.state.run_cost,
                                 self.cost.state.ratio, self.cost.state.downgraded)
            if outcome.action == "stop":
                state.answer = (
                    "抱歉，本次会话的处理预算已用完，为避免继续产生费用我先停止。\n"
                    "已为你创建工单并转人工专员继续处理。"
                )
                state.handover = True
                return Command(goto="finalize")

        self.ex._record_step(state, "llm_call", {
            "text": (reply.text or "")[:500],
            "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls],
            "model": reply.model or model,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "downgraded": bool(self.cost and self.cost.state.downgraded),
        }, reply.duration_ms)

        if self.tracer is not None:
            self.tracer.llm_call(reply.model or model, reply.prompt_tokens,
                                 reply.completion_tokens, reply.duration_ms,
                                 reply.text or "", len(reply.tool_calls),
                                 0.0, bool(self.cost and self.cost.state.downgraded))

        state.messages.append(_assistant_message(reply))

        if not reply.tool_calls:
            state.answer = (reply.text or "").strip() or "（模型没有给出内容）"
            return Command(goto="finalize")

        # 按顺序执行工具；遇到需要审批的，就地挂起（保证 tool 消息顺序与 tool_calls 一致）
        for call in reply.tool_calls:
            result = self.ex.call(state, call.name, call.arguments, call_id=call.id)
            if result.requires_approval:
                state.pending_approval = {
                    "name": call.name,
                    "arguments": call.arguments,
                    "call_id": call.id,
                }
                return Command(goto="approval")
            state.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": result.to_llm_content(),
            })

        return Command(goto="agent")     # 继续 ReAct 循环

    def node_approval(self, state: AgentState):
        pending = state.pending_approval or {}
        update = self.kit.hitl.gate(
            state,
            action=pending.get("name", "unknown"),
            arguments=pending.get("arguments", {}),
            reason=f"自主 Agent 请求执行关键动作：{pending.get('name')}",
            risk_level="high",
            amount=float(pending.get("arguments", {}).get("amount", 0) or 0),
        )
        decision = (update.get("facts") or {}).get("_approval", {})
        if self.tracer is not None:
            self.tracer.approval(
                decision.get("approval_id", ""), pending.get("name", ""),
                "approved" if decision.get("approved") else "rejected",
                float(pending.get("arguments", {}).get("amount", 0) or 0),
                decision.get("note", ""),
            )
        # 待执行的调用要连同审批结果一起放进 Command.update ——
        # 因为 gate 返回的 update 里 pending_approval 是 None，
        # 直接改 state 属性会被随后的合并覆盖掉。
        update["pending_approval"] = {**pending, "_approved": bool(decision.get("approved"))}
        target = "execute_approved" if decision.get("approved") else "denied"
        return Command(goto=target, update=update)

    def node_execute_approved(self, state: AgentState):
        pending = state.pending_approval or {}
        result = self.ex.call(
            state, pending.get("name", ""), pending.get("arguments", {}) or {},
            approved=True, call_id=pending.get("call_id", ""),
        )
        state.messages.append({
            "role": "tool",
            "tool_call_id": pending.get("call_id", ""),
            "name": pending.get("name", ""),
            "content": result.to_llm_content(),
        })
        state.pending_approval = None
        return Command(goto="agent")

    def node_denied(self, state: AgentState):
        pending = state.pending_approval or {}
        state.messages.append({
            "role": "tool",
            "tool_call_id": pending.get("call_id", ""),
            "name": pending.get("name", ""),
            "content": (
                "[人工审批未通过] 该关键动作已被客服驳回，未执行。"
                "请不要重复提交同样的申请；改为向客户说明情况，并转人工专员复核。"
            ),
        })
        state.pending_approval = None
        state.approval_denied = True
        return Command(goto="agent")

    def node_finalize(self, state: AgentState):
        ticket_id = state.fact("ticket_id") or ""
        needs_human = state.handover or state.approval_denied
        if needs_human and not ticket_id:
            self.kit.ensure_ticket(state)
        update = self.kit.finish(
            state,
            (state.answer or "")[:80] or "自主 Agent 已答复",
            handover=bool(state.handover),
        )
        return update

    # ------------------------------------------------------------ 建图
    def build(self) -> StateGraph:
        graph = StateGraph("long_tail_agent")
        graph.add_node("agent", self.node_agent, fallback="agent_fallback")
        graph.add_node("approval", self.node_approval)
        graph.add_node("execute_approved", self.node_execute_approved)
        graph.add_node("denied", self.node_denied)
        graph.add_node("agent_fallback", self.node_agent_fallback)
        graph.add_node("finalize", self.node_finalize)
        graph.set_entry_point("agent")
        graph.add_edge("approval", "execute_approved")   # 出边由 Command 决定
        graph.add_edge("execute_approved", "agent")
        graph.add_edge("denied", "agent")
        graph.add_edge("agent_fallback", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def node_agent_fallback(self, state: AgentState):
        """失败回退：自主 Agent 反复失败时不硬撑，建单转人工。"""
        self.kit.ensure_ticket(state)
        state.answer = (
            "抱歉，这个问题的自动处理没有成功，我已为你登记工单并转人工专员跟进，"
            "专员会在 24 小时内联系你。"
        )
        state.handover = True
        return {}


def _assistant_message(reply: Any) -> dict:
    message: dict = {"role": "assistant", "content": reply.text or ""}
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments
                    or json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in reply.tool_calls
        ]
    return message
