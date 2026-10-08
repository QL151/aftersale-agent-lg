"""状态机引擎测试：循环上限、失败回退、断点续跑、人机协同原语。

这是整个项目最需要测试的地方 —— 因为它的 bug 不会崩，而是"卡住"或"重复执行"，
线上表现为"工单莫名停在待审批"或"客户被退两次款"。
"""

from __future__ import annotations

import pytest

from app.graph import (
    END,
    Command,
    FileCheckpointer,
    MemoryCheckpointer,
    RetryPolicy,
    StateGraph,
    build_checkpointer,
    build_engine,
    interrupt,
)
from app.schemas import AgentState


def mk_state(task="t", thread="th"):
    return AgentState(run_id="r-test", thread_id=thread, task=task)


class TestLinearFlow:
    def test_edges_and_conditional(self):
        g = StateGraph("linear")
        g.add_node("a", lambda s: {"answer": "A"})
        g.add_node("b", lambda s: {"answer": s.answer + "B"})
        g.add_node("c", lambda s: {"answer": s.answer + "C"})
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_conditional_edges("b", lambda s: "go" if len(s.answer) > 1 else "stop",
                                {"go": "c", "stop": END})
        g.add_edge("c", END)
        out = g.compile().invoke(mk_state())
        assert out.state.answer == "ABC"
        assert out.stopped_reason == "finished"

    def test_conditional_without_path_map(self):
        """条件函数直接返回节点名时必须可用（工作流就是靠这个）。"""
        g = StateGraph("nomap")
        g.add_node("start", lambda s: {})
        g.add_node("left", lambda s: {"answer": "L"})
        g.add_node("right", lambda s: {"answer": "R"})
        g.set_entry_point("start")
        g.add_conditional_edges("start", lambda s: "left" if "x" in s.task else "right")
        g.add_edge("left", END)
        g.add_edge("right", END)
        assert g.compile().invoke(mk_state("x")).state.answer == "L"
        assert g.compile().invoke(mk_state("y")).state.answer == "R"

    def test_command_jump(self):
        g = StateGraph("cmd")
        g.add_node("a", lambda s: Command(goto="c", update={"answer": "A"}))
        g.add_node("b", lambda s: {"answer": "不该走到这里"})
        g.add_node("c", lambda s: {"answer": s.answer + "C"})
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_edge("c", END)
        assert g.compile().invoke(mk_state()).state.answer == "AC"


class TestLoopGuard:
    def test_recursion_limit_stops_loop(self):
        g = StateGraph("loop")
        g.add_node("spin", lambda s: {"tool_calls": s.tool_calls + 1})
        g.set_entry_point("spin")
        g.add_edge("spin", "spin")
        out = g.compile(recursion_limit=5).invoke(mk_state())
        assert out.stopped_reason == "max_steps"
        assert out.state.tool_calls == 5
        assert out.nodes_executed == 5

    def test_time_limit(self):
        g = StateGraph("slow")
        g.add_node("slow", lambda s: {"tool_calls": s.tool_calls + 1})
        g.set_entry_point("slow")
        g.add_edge("slow", "slow")
        out = g.compile(recursion_limit=10_000, max_seconds=0.0).invoke(mk_state())
        assert out.stopped_reason == "timeout"

    def test_missing_node_raises_at_compile(self):
        g = StateGraph("bad")
        g.add_node("a", lambda s: {})
        with pytest.raises(Exception):
            g.compile()          # 没设 entry point


class TestFailureFallback:
    def test_retry_policy_recovers(self):
        attempts = {"n": 0}

        def flaky(s):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError("boom")
            return {"answer": "recovered"}

        g = StateGraph("retry")
        g.add_node("flaky", flaky, retry=RetryPolicy(max_attempts=3))
        g.set_entry_point("flaky")
        g.add_edge("flaky", END)
        out = g.compile().invoke(mk_state())
        assert out.state.answer == "recovered"
        assert attempts["n"] == 3

    def test_fallback_node_on_exhausted_retry(self):
        g = StateGraph("fallback")
        g.add_node("bad", lambda s: (_ for _ in ()).throw(RuntimeError("always fails")),
                   retry=RetryPolicy(max_attempts=2), fallback="human")
        g.add_node("human", lambda s: {"answer": "转人工", "handover": True})
        g.set_entry_point("bad")
        g.add_edge("human", END)
        out = g.compile().invoke(mk_state())
        assert out.state.answer == "转人工"
        assert out.state.handover is True

    def test_no_fallback_surfaces_error(self):
        g = StateGraph("no_fallback")
        g.add_node("bad", lambda s: (_ for _ in ()).throw(RuntimeError("nope")))
        g.set_entry_point("bad")
        g.add_edge("bad", END)
        out = g.compile().invoke(mk_state())
        assert out.stopped_reason == "error"
        assert "nope" in out.error


class TestInterruptResume:
    def build(self, checkpointer):
        def need_approval(s):
            decision = interrupt({"action": "issue_refund", "amount": 199.0})
            return {
                "approvals": s.approvals + 1,
                "answer": f"审批结果={decision}",
                "facts": {**s.facts, "decision": decision},
            }

        g = StateGraph("hitl")
        g.add_node("gate", need_approval)
        g.set_entry_point("gate")
        g.add_edge("gate", END)
        return g.compile(checkpointer=checkpointer, recursion_limit=5)

    def test_interrupt_pauses_with_payload(self):
        cp = MemoryCheckpointer()
        out = self.build(cp).invoke(mk_state(thread="t1"), thread_id="t1")
        assert out.interrupted
        assert out.stopped_reason == "interrupted"
        assert out.pending == {"action": "issue_refund", "amount": 199.0}
        assert cp.load("t1") is not None          # 状态必须落盘

    def test_resume_continues(self):
        cp = MemoryCheckpointer()
        graph = self.build(cp)
        graph.invoke(mk_state(thread="t2"), thread_id="t2")
        out = graph.resume("approved", thread_id="t2")
        assert out.state.answer == "审批结果=approved"
        assert out.state.approvals == 1

    def test_resume_from_fresh_instance(self):
        """换一个进程/实例也能恢复 —— 这是"审批可能发生在几小时后"的前提。"""
        cp = MemoryCheckpointer()
        self.build(cp).invoke(mk_state(thread="t3"), thread_id="t3")
        fresh = self.build(cp)
        out = fresh.resume("rejected", thread_id="t3")
        assert out.state.answer == "审批结果=rejected"

    def test_resume_without_checkpoint_errors(self):
        graph = self.build(None)
        with pytest.raises(Exception):
            graph.resume("x", thread_id="nope")

    def test_multiple_interrupts_in_order(self):
        """一个节点里两次 interrupt：恢复时要按顺序各接一个值。"""
        def two_gates(s):
            first = interrupt("first")
            second = interrupt("second")
            return {"answer": f"{first}/{second}"}

        g = StateGraph("two")
        g.add_node("g", two_gates)
        g.set_entry_point("g")
        g.add_edge("g", END)
        cp = MemoryCheckpointer()
        graph = g.compile(checkpointer=cp)
        assert graph.invoke(mk_state(thread="t4"), thread_id="t4").pending == "first"
        out = graph.resume(["A", "B"], thread_id="t4")
        assert out.state.answer == "A/B"


class TestCheckpointers:
    def test_file_checkpointer_roundtrip(self, tmp_path):
        cp = FileCheckpointer(tmp_path / "ckpt")
        cp.save("thread-x", {"state": {"run_id": "r"}, "next_node": "n"})
        assert cp.load("thread-x")["next_node"] == "n"
        assert "thread-x" in cp.list_threads()
        cp.delete("thread-x")
        assert cp.load("thread-x") is None

    def test_file_checkpointer_sanitizes_name(self, tmp_path):
        """会话 ID 里带路径分隔符不能变成目录穿越。"""
        cp = FileCheckpointer(tmp_path / "ckpt")
        cp.save("../../evil", {"ok": True})
        assert cp.load("../../evil") == {"ok": True}
        assert not (tmp_path / "evil.json").exists()

    def test_build_checkpointer_returns_usable(self):
        assert getattr(build_checkpointer(), "name", "") in {"file", "chain", "redis", "memory"}


class TestEngineSelection:
    def test_auto_falls_back_to_native(self):
        g = StateGraph("e")
        g.add_node("a", lambda s: {})
        g.set_entry_point("a")
        g.add_edge("a", END)
        compiled = build_engine(g, force="auto")
        assert compiled.engine in {"native", "langgraph"}

    def test_forced_native(self):
        g = StateGraph("e2")
        g.add_node("a", lambda s: {})
        g.set_entry_point("a")
        g.add_edge("a", END)
        assert build_engine(g, force="native").engine == "native"

    def test_forced_langgraph_degrades_gracefully(self):
        """没装 langgraph 时不能崩，要回退并让人知道。"""
        g = StateGraph("e3")
        g.add_node("a", lambda s: {})
        g.set_entry_point("a")
        g.add_edge("a", END)
        compiled = build_engine(g, force="langgraph")
        assert compiled.engine in {"native", "langgraph"}

    def test_state_payload_roundtrip(self):
        """状态必须能完整存档再还原（含嵌套的 Step 对象）。"""
        from app.schemas import Step

        state = mk_state()
        state.steps.append(Step(1, "tool_call", 12, {"tool": "x", "ok": True}))
        state.facts["nested"] = {"a": [1, 2, {"b": "c"}]}
        state.put("order", {"order_id": "SO1"})
        restored = AgentState.from_payload(state.to_payload())
        assert restored.run_id == state.run_id
        assert restored.steps[0].type == "tool_call"
        assert restored.steps[0].detail["tool"] == "x"
        assert restored.facts["order"]["order_id"] == "SO1"
        assert restored.facts["nested"]["a"][2]["b"] == "c"
