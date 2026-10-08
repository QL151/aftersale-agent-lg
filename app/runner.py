"""售后 Agent 门面：把路由 / 状态机 / 记忆 / 成本 / 观测 / 人机协同串成一次运行。

一次 run 的完整生命周期（也是面试讲架构时最该讲清楚的一段）：

  ① 载入会话记忆（Redis / 本地）→ 拿到历史槽位，避免重复追问
  ② 意图路由：规则打分 → （必要时）模型分类 → 判定「确定性工作流」还是「长尾自主 Agent」
  ③ 按路由结果编译对应的图（8 条工作流之一，或 ReAct Agent 图）
  ④ 执行图：每个节点经过 Guardrail；关键动作在**独立节点**里 interrupt 等人工审批
  ⑤ 落盘：Trace（本地 JSONL + 可选 Langfuse）、Checkpoint、会话记忆、长期偏好
  ⑥ 汇总成 RunResult（含 token / 成本 / 是否降级 / 是否全自动闭环 / 引擎）

  其中 ④ 的"中断"会让这个函数**直接返回**（stopped_reason=interrupted），
  等人工决策后再调 resume() —— 可以换一个进程、换一台机器。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from .agent_graph import AgentGraphBuilder
from .config import Settings, settings as global_settings
from .cost import CostController, session_usage
from .db import Repo
from .executor import ToolExecutor
from .graph.checkpoint import build_checkpointer
from .graph.engine import build_engine
from .guardrail import Guardrail
from .hitl import ApprovalManager
from .llm import build_llm
from .memory.profile import profile_service
from .memory.session import session_service
from .obs.tracer import Tracer, load_trace, new_run_id
from .router import IntentRouter
from .schemas import AgentState, RunResult, Step
from .tools import load_all, specs_for
from .workflows import WorkflowKit, build_workflow, workflow_names

EventCallback = Callable[[str, dict], None]


class AftersaleAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: Any = None,
        repo: Repo | None = None,
        on_event: EventCallback | None = None,
        engine: str | None = None,
    ) -> None:
        self.s = settings or global_settings
        load_all()
        self.s.ensure_dirs()
        self.repo = repo or Repo()
        self.llm = llm or build_llm(self.s)
        self.guardrail = Guardrail(self.s)
        self.router = IntentRouter(self.llm)
        self.hitl = ApprovalManager(self.repo)
        self.memory = session_service()
        self.profile = profile_service()
        self.on_event = on_event
        self.engine = engine            # 强制指定引擎（CLI/测试用）
        self._graphs: dict[str, Any] = {}

    # ================================================================ 主流程
    def run(
        self,
        task: str,
        session_id: str = "",
        thread_id: str = "",
        slots: dict[str, Any] | None = None,
    ) -> RunResult:
        run_id = new_run_id()
        thread_id = thread_id or run_id
        session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"

        tracer = Tracer(run_id, task, session_id)
        self._emit("run_start", {"run_id": run_id, "thread_id": thread_id,
                                 "session_id": session_id, "task": task})

        # ---------- ① 会话记忆 ----------
        memory = self.memory.load(session_id)
        memory_turns = len(memory.turns)
        session_tokens, session_cost = session_usage(self.memory, session_id)
        self.memory.append_turn(session_id, "user", task)

        state = AgentState(
            run_id=run_id, thread_id=thread_id, task=task, session_id=session_id,
            slots={**(memory.slots or {}), **(slots or {})},
        )

        # ---------- ② 意图路由 ----------
        decision = self.router.route(task, slots=slots, memory_slots=memory.slots)
        state.intent = decision.intent
        state.route = decision.route
        state.workflow = decision.workflow
        state.confidence = decision.confidence
        state.slots = decision.slots
        state.steps.append(Step(1, "route", 0, {
            "intent": decision.intent, "label": decision.label,
            "confidence": decision.confidence, "route": decision.route,
            "workflow": decision.workflow, "source": decision.source,
            "reason": decision.reason,
        }))
        tracer.event("route", "intent_router", {
            "intent": decision.intent, "label": decision.label,
            "confidence": decision.confidence, "route": decision.route,
            "workflow": decision.workflow, "source": decision.source,
            "slots": decision.slots,
        })
        self._emit("route", decision.to_dict())

        # ---------- ③ 编译图 ----------
        cost = CostController(session_id, self.memory, session_tokens, session_cost)
        executor = ToolExecutor(self.s, self.guardrail, tracer, self.repo,
                                self.profile, self.memory)
        kit = WorkflowKit(executor, self.repo, tracer, self.hitl, self.s)

        graph_obj, graph_name = self._build_graph(decision, executor, kit, cost, tracer)
        compiled = self._compile(graph_name, graph_obj, tracer)

        # ---------- ④ 执行 ----------
        started = time.time()
        outcome = compiled.invoke(state, thread_id=thread_id)
        final_state: AgentState = outcome.state
        total_ms = int((time.time() - started) * 1000)

        stopped = outcome.stopped_reason
        if outcome.interrupted:
            stopped = "interrupted"
            pending = outcome.pending or {}
            final_state.answer = (
                "这个操作需要人工审核，我已经提交给客服专员了。\n"
                f"待审核动作：{pending.get('action', '关键动作')}"
                f"（金额 {pending.get('amount', 0)} 元）\n"
                f"审核原因：{pending.get('reason', '')}\n"
                "审核通过后会立即为你处理，通常 30 分钟内完成，请留意通知。"
            )
        elif stopped == "max_steps":
            final_state.answer = final_state.answer or (
                f"处理步骤超过 {self.s.max_steps} 步仍未完成，已停止并转人工。"
            )
            final_state.handover = True
            self._safe_escalate(final_state, "自动处理步数超限")
        elif stopped == "timeout":
            final_state.answer = final_state.answer or "处理超时，已转人工跟进。"
            final_state.handover = True
            self._safe_escalate(final_state, "自动处理超时")
        elif stopped == "error":
            final_state.answer = final_state.answer or (
                f"处理过程中出现异常（{outcome.error}），已转人工跟进。"
            )
            final_state.handover = True
            self._safe_escalate(final_state, f"处理异常：{outcome.error[:60]}")

        if final_state.needs_clarify:
            stopped = "clarify"

        if not final_state.answer:
            final_state.answer = "（没有产生结论）"

        # ---------- ⑤ 落盘 ----------
        self.memory.append_turn(session_id, "assistant", final_state.answer[:1000])
        self.memory.remember_slots(
            session_id,
            **{k: v for k, v in (final_state.slots or {}).items()
               if v and not k.startswith("_")},
        )
        # 追问未完成时记住意图：下一轮用户只回"SO2026xxxxx"也必须能接上，
        # 否则多轮澄清会丢意图、被迫重新开始 —— 这是交互轮次居高不下的主因。
        if stopped == "clarify":
            self.memory.remember_slots(session_id, _pending_intent=final_state.intent)
        else:
            self.memory.forget(session_id, "_pending_intent")

        result = self._to_result(final_state, stopped, total_ms, tracer, outcome, compiled.engine)
        tracer.metadata.update({
            "engine": compiled.engine, "session_turns": memory_turns + 2,
            "slots": final_state.slots, "cost": cost.state.to_dict(),
        })
        tracer.finish(result)

        self._emit("run_end", result.to_dict())
        return result

    # ================================================================ 恢复
    def resume(self, thread_id: str, value: Any = None) -> RunResult:
        """人工决策后恢复被中断的图。可在**另一个进程/机器**上执行。"""
        checkpointer = build_checkpointer()
        saved = checkpointer.load(thread_id)
        if not saved:
            raise ValueError(f"没有找到 thread {thread_id} 的执行存档，无法恢复")

        graph_name = saved.get("graph", "")
        state = AgentState.from_payload(saved["state"])
        prior = load_trace(state.run_id) or {}
        tracer = Tracer(state.run_id, state.task, state.session_id, prior=prior)

        executor = ToolExecutor(self.s, self.guardrail, tracer, self.repo,
                                self.profile, self.memory)
        kit = WorkflowKit(executor, self.repo, tracer, self.hitl, self.s)
        graph_obj = self._build_graph_by_name(graph_name, executor, kit, None, tracer)
        if graph_obj is None:
            raise ValueError(f"存档里的图 {graph_name} 已不存在")

        compiled = self._compile(graph_name, graph_obj, tracer, checkpointer=checkpointer)
        started = time.time()
        outcome = compiled.resume(value, thread_id=thread_id)
        final_state = outcome.state or state
        total_ms = int((time.time() - started) * 1000)

        if not final_state.answer:
            final_state.answer = "（恢复后没有产生结论）"

        self.memory.append_turn(final_state.session_id, "assistant",
                                final_state.answer[:1000])
        self.memory.remember_slots(
            final_state.session_id,
            **{k: v for k, v in (final_state.slots or {}).items() if v},
        )

        result = self._to_result(final_state, outcome.stopped_reason, total_ms, tracer, outcome,
                                 compiled.engine)
        tracer.metadata.update({"resumed": True, "engine": compiled.engine})
        tracer.finish(result)
        self._emit("run_end", result.to_dict())
        return result

    # ================================================================ 图构建
    def _build_graph(self, decision: Any, executor: Any, kit: Any, cost: Any, tracer: Any):
        if decision.route == "workflow" and decision.workflow in workflow_names():
            graph = build_workflow(decision.workflow, kit)
            if graph is not None:
                return graph, decision.workflow
        return self._agent_graph(executor, kit, cost, tracer), "long_tail_agent"

    def _agent_graph(self, executor: Any, kit: Any, cost: Any, tracer: Any):
        builder = AgentGraphBuilder(
            llm=self.llm, executor=executor, kit=kit, tools_spec=specs_for(),
            cost=cost, tracer=tracer, settings=self.s,
        )
        return builder.build()

    def _build_graph_by_name(self, name: str, executor: Any, kit: Any, cost: Any, tracer: Any):
        if name == "long_tail_agent":
            return self._agent_graph(executor, kit, cost, tracer)
        return build_workflow(name, kit)

    def _compile(self, name: str, graph_obj: Any, tracer: Any, checkpointer: Any = None):
        emitted = {"count": 0}

        def on_node(node_name: str, state: Any, duration_ms: int) -> None:
            tracer.node(node_name, duration_ms, {"node": node_name})
            self._emit("node", {"node": node_name, "duration_ms": duration_ms,
                                "run_id": getattr(state, "run_id", "")})
            # 把这一节点内产生的工具调用/拦截/审批步骤也推给前端，
            # 否则 SSE 只能看到"节点完成"，看不到"调了哪个工具、有没有被拦"。
            steps = getattr(state, "steps", []) or []
            for step in steps[emitted["count"]:]:
                self._emit("step", step.to_dict())
            emitted["count"] = len(steps)

        return build_engine(
            graph_obj,
            checkpointer=checkpointer or build_checkpointer(),
            recursion_limit=self.s.max_steps,
            max_seconds=self.s.max_total_seconds,
            on_node=on_node,
            force=self.engine,
        )

    # ================================================================ 辅助
    def _to_result(self, state: AgentState, stopped: str, total_ms: int,
                   tracer: Tracer, outcome: Any, engine: str = "native") -> RunResult:
        return RunResult(
            run_id=state.run_id,
            thread_id=state.thread_id,
            task=state.task,
            answer=state.answer,
            intent=state.intent,
            route=state.route,
            workflow=state.workflow,
            steps=list(state.steps),
            stopped_reason=stopped,
            total_ms=total_ms,
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            cost_yuan=state.cost_yuan,
            model_used=state.model_used or ("mock" if self.s.use_mock else self.s.llm_model),
            downgraded=state.downgraded,
            handover=state.handover,
            auto_closed=state.auto_closed,
            approvals=state.approvals,
            engine=engine,
        )

    def _safe_escalate(self, state: AgentState, reason: str) -> None:
        """兜底转人工：任何异常路径都不能让客户的问题"消失"。"""
        try:
            if not state.fact("ticket_id"):
                order = state.fact("order") or {}
                order_id = order.get("order_id") or state.slots.get("order_id") or ""
                if order_id:
                    text = self.repo.create_ticket(
                        order_id, state.fact("customer_id", ""), state.intent or "售后",
                        f"{state.intent or '售后'}：{state.task[:40]}",
                    )
                    state.put("ticket_id", text.ticket_id)
            ticket_id = state.fact("ticket_id")
            if ticket_id:
                self.repo.update_ticket(ticket_id, status="待人工", assignee="人工专员-兜底")
                self.repo.add_note(ticket_id, "系统", f"自动流程兜底升级：{reason}")
        except Exception:  # noqa: BLE001
            pass

    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event:
            try:
                self.on_event(kind, payload)
            except Exception:  # noqa: BLE001
                pass

    # ================================================================ 审批
    def pending_approvals(self) -> list[dict]:
        return self.hitl.pending()

    def decide_approval(self, approval_id: str, approved: bool, by: str = "人工客服",
                        note: str = "") -> dict:
        """审批：先落库（唯一真相），再唤醒对应的图。"""
        record = self.hitl.get(approval_id)
        if not record:
            return {"ok": False, "error": f"审批单 {approval_id} 不存在"}
        ok = self.hitl.decide(approval_id, approved, by, note)
        if not ok:
            return {"ok": False, "error": f"审批单 {approval_id} 已被处理过"}

        thread_id = record["thread_id"]
        try:
            result = self.resume(thread_id, {"approved": approved, "by": by, "note": note})
        except Exception as exc:  # noqa: BLE001
            return {"ok": True, "resumed": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "resumed": True, "result": result.to_dict()}


def explain_run(run_id: str) -> str:
    from .obs.badcase import explain

    return explain(run_id)


def load_run(run_id: str) -> dict | None:
    return load_trace(run_id)
