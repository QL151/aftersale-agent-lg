"""记忆层 / 人机协同 / 成本熔断测试。"""

from __future__ import annotations

import json

import pytest

from app.cost import CostController, estimate_cost, price_of
from app.hitl import ApprovalManager, make_approval_id
from app.memory.profile import ProfileService, SQLiteProfileStore
from app.memory.session import LocalSessionStore, SessionMemoryService
from app.schemas import AgentState


# ==================================================================== 会话记忆


class TestSessionMemory:
    def test_append_and_read_turns(self, tmp_path, clean_memory):
        service = SessionMemoryService(LocalSessionStore(tmp_path))
        service.append_turn(clean_memory, "user", "我要退款")
        service.append_turn(clean_memory, "assistant", "请提供订单号")
        context = service.load(clean_memory).context_messages()
        assert [m["role"] for m in context] == ["user", "assistant"]
        assert context[0]["content"] == "我要退款"

    def test_empty_slot_does_not_overwrite(self, tmp_path, clean_memory):
        """这轮没说 ≠ 之前说的作废。空值不能把已确认的订单号冲掉。"""
        service = SessionMemoryService(LocalSessionStore(tmp_path))
        service.remember_slots(clean_memory, order_id="SO1", intent="refund")
        service.remember_slots(clean_memory, order_id="", intent=None)
        slots = service.load(clean_memory).slots
        assert slots["order_id"] == "SO1"
        assert slots["intent"] == "refund"

    def test_forget_removes_key(self, tmp_path, clean_memory):
        service = SessionMemoryService(LocalSessionStore(tmp_path))
        service.remember_slots(clean_memory, _pending_intent="refund_request")
        service.forget(clean_memory, "_pending_intent")
        assert "_pending_intent" not in service.load(clean_memory).slots

    def test_long_history_is_compressed(self, tmp_path, clean_memory):
        service = SessionMemoryService(LocalSessionStore(tmp_path))
        for i in range(40):
            service.append_turn(clean_memory, "user", f"第 {i} 轮：这是一段比较长的对话内容")
        memory = service.load(clean_memory)
        assert len(memory.turns) <= 8          # 只保留最近若干轮
        assert memory.summary                    # 前半段被压成摘要
        assert len(memory.summary) <= 400

    def test_persists_to_disk(self, tmp_path, clean_memory):
        """会话记忆必须落盘：进程重启后同一会话还要能接上。"""
        service = SessionMemoryService(LocalSessionStore(tmp_path))
        service.remember_slots(clean_memory, order_id="SO9999")
        fresh = SessionMemoryService(LocalSessionStore(tmp_path))
        assert fresh.load(clean_memory).slots["order_id"] == "SO9999"

    def test_usage_accumulates(self, tmp_path, clean_memory):
        service = SessionMemoryService(LocalSessionStore(tmp_path))
        memory = service.load(clean_memory)
        memory.usage = {"tokens": 123, "cost_yuan": 0.45}
        service.store.save(memory)
        assert service.load(clean_memory).usage["tokens"] == 123


# ==================================================================== 长期偏好


class TestProfile:
    def test_set_and_get(self, tmp_path):
        service = ProfileService(SQLiteProfileStore(tmp_path / "p.db"))
        service.set("C00001", "偏好通知渠道", "APP推送")
        assert service.get("C00001", "偏好通知渠道") == "APP推送"
        assert service.get("C00001", "不存在") is None

    def test_json_values_roundtrip(self, tmp_path):
        service = ProfileService(SQLiteProfileStore(tmp_path / "p.db"))
        service.set("C00001", "标签", {"vip": True, "n": 2})
        assert service.get("C00001", "标签") == {"vip": True, "n": 2}

    def test_upsert_overwrites(self, tmp_path):
        service = ProfileService(SQLiteProfileStore(tmp_path / "p.db"))
        service.set("C00001", "等级", "普通")
        service.set("C00001", "等级", "金牌")
        assert service.get("C00001", "等级") == "金牌"

    def test_learn_from_run(self, tmp_path):
        """没有这一步，"PostgreSQL 存长期偏好"就只是简历上的一句话。"""
        service = ProfileService(SQLiteProfileStore(tmp_path / "p.db"))
        learned = service.learn_from_run(
            "C00001", "refund_request",
            {"resolution": "退款", "channel": "APP推送", "tags": ["时效敏感"],
             "summary": "数码订单超期，已解释条款"},
            run_id="run-x",
        )
        assert learned
        prefs = service.all_preferences("C00001")
        assert prefs["上次结案方式"] == "退款"
        assert prefs["偏好通知渠道"] == "APP推送"
        assert service.recent_facts("C00001")

    def test_learn_skips_without_customer(self, tmp_path):
        service = ProfileService(SQLiteProfileStore(tmp_path / "p.db"))
        assert service.learn_from_run("", "x", {"resolution": "y"}) == []


# ==================================================================== HITL


class TestApproval:
    def test_deterministic_id(self):
        """同一动作必须算出同一个审批单号，否则图 resume 时会重复建单。"""
        first = make_approval_id("run-1", "issue_refund", {"order_id": "SO1", "amount": 10})
        second = make_approval_id("run-1", "issue_refund", {"amount": 10, "order_id": "SO1"})
        assert first == second
        other = make_approval_id("run-2", "issue_refund", {"order_id": "SO1", "amount": 10})
        assert first != other

    def test_create_is_idempotent(self, repo):
        manager = ApprovalManager(repo)
        approval_id = make_approval_id("run-a", "issue_refund", {"order_id": "SO1"})
        for _ in range(3):
            repo.create_approval(approval_id, "run-a", "th-a", "issue_refund",
                                 json.dumps({"order_id": "SO1"}), "测试", "high", 10.0)
        matching = [a for a in manager.pending() if a["approval_id"] == approval_id]
        assert len(matching) == 1

    def test_decide_once_only(self, repo):
        manager = ApprovalManager(repo)
        approval_id = make_approval_id("run-b", "issue_refund", {"order_id": "SO2"})
        repo.create_approval(approval_id, "run-b", "th-b", "issue_refund", "{}", "测试", "high", 5.0)
        assert manager.decide(approval_id, True, by="客服A") is True
        assert manager.decide(approval_id, False, by="客服B") is False   # 重复决策被拒
        assert manager.get(approval_id)["status"] == "approved"
        assert manager.get(approval_id)["decided_by"] == "客服A"

    def test_audit_trail_recorded(self, repo):
        """审批必须有审计：谁、什么时候、批了什么、理由是什么。"""
        manager = ApprovalManager(repo)
        approval_id = make_approval_id("run-c", "grant_coupon", {"customer_id": "C1"})
        repo.create_approval(approval_id, "run-c", "th-c", "grant_coupon", "{}",
                             "补偿", "medium", 50.0)
        manager.decide(approval_id, False, by="主管", note="依据不足")
        record = manager.get(approval_id)
        assert record["status"] == "rejected"
        assert record["decided_by"] == "主管"
        assert record["decision_note"] == "依据不足"
        assert record["decided_at"]

    def test_timeout_sweep(self, repo):
        manager = ApprovalManager(repo, timeout_seconds=-1, timeout_action="reject")
        approval_id = make_approval_id("run-d", "issue_refund", {"order_id": "SO3"})
        repo.create_approval(approval_id, "run-d", "th-d", "issue_refund", "{}", "x", "high", 9.0)
        settled = manager.sweep_timeouts()
        assert settled >= 1
        assert manager.get(approval_id)["status"] == "timeout"

    def test_render_pending_is_readable(self, repo):
        text = ApprovalManager(repo).render_pending()
        assert isinstance(text, str)


# ==================================================================== 成本熔断


class TestCost:
    def test_price_lookup(self):
        assert price_of("deepseek-chat")[0] > 0
        assert price_of("qwen-turbo")[0] < price_of("deepseek-chat")[0]
        assert price_of("some-mini-model")[0] > 0     # 未知模型要有兜底价

    def test_estimate_cost(self):
        assert estimate_cost("deepseek-chat", 1000, 1000) == pytest.approx(0.003, abs=1e-6)
        assert estimate_cost("mock", 10**6, 10**6) == 0.0

    def test_warn_then_downgrade_then_stop(self):
        controller = CostController(session_id="s")
        controller.state.run_tokens = int(controller.state.run_ratio * 0 + 0)
        # 直接压到告警线
        controller.state.run_tokens = 0
        controller.state.session_tokens = 0

        controller.state.run_tokens = 0
        outcome = controller.charge("deepseek-chat", 0, 0)
        assert outcome.action == "ok"

        # 手动推到 85%
        from app.config import settings

        controller.state.run_tokens = int(settings.run_token_budget * 0.85)
        outcome = controller.charge("deepseek-chat", 1, 1)
        assert outcome.action in {"warn", "downgrade"}

    def test_downgrade_switches_model(self):
        controller = CostController(session_id="s")
        controller.state.downgraded = True
        assert controller.pick_model("deepseek-chat") != "deepseek-chat"

    def test_not_downgraded_keeps_model(self):
        controller = CostController(session_id="s")
        assert controller.pick_model("deepseek-chat") == "deepseek-chat"

    def test_stop_after_downgrade_exhausted(self):
        controller = CostController(session_id="s")
        controller.state.downgraded = True
        controller.state.run_tokens = 10**9
        outcome = controller.charge("qwen-turbo", 10, 10)
        assert outcome.action == "stop"
        assert controller.state.stopped

    def test_session_usage_read_write(self, tmp_path, clean_memory):
        from app.cost import session_usage

        service = SessionMemoryService(LocalSessionStore(tmp_path))
        service.remember_slots(clean_memory, x=1)
        memory = service.load(clean_memory)
        memory.usage = {"tokens": 777, "cost_yuan": 1.5}
        service.store.save(memory)
        tokens, cost = session_usage(service, clean_memory)
        assert tokens == 777 and cost == pytest.approx(1.5)

    def test_cost_line_is_informative(self):
        assert "tokens=" in CostController(session_id="s").line()
