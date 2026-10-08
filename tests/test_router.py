"""意图路由测试。

路由层是"省 token 且更稳"这个说法的兑现处：
它错一次，要么高频意图被拖进昂贵的自主 Agent，要么客户诉求被派给错误的流程。
"""

from __future__ import annotations

import pytest

from app.router import INTENTS, IntentRouter, describe_routing, extract_slots
from app.workflows import workflow_names


def route(text, **kwargs):
    """测试助手：把构造参数和 route() 参数分开传。"""
    router_keys = {"threshold", "use_llm", "llm"}
    router = IntentRouter(llm=kwargs.pop("llm", None),
                          **{k: v for k, v in kwargs.items() if k in router_keys})
    return router.route(text, **{k: v for k, v in kwargs.items() if k not in router_keys})


class TestHighFrequencyIntents:
    @pytest.mark.parametrize("text,intent", [
        ("帮我查下订单 SO2026000001 的物流到哪了", "logistics_query"),
        ("我的快递三天没动了", "logistics_query"),
        ("订单什么时候能到", "logistics_query"),
        ("我要退款，订单号 SO2026000001", "refund_request"),
        ("这个订单不想要了，帮我退了吧", "refund_request"),
        ("我的退款什么时候到账", "refund_progress"),
        ("退款进度查一下", "refund_progress"),
        ("我要退货，尺码不对", "return_exchange"),
        ("能换个颜色吗", "return_exchange"),
        ("我要开发票，公司抬头", "invoice_issue"),
        ("发票抬头写错了要重开", "invoice_issue"),
        ("刚买就降价了，能补差价吗", "price_protection"),
        ("我要申请价保", "price_protection"),
        ("我的优惠券为什么用不了", "coupon_promo"),
        ("这个满减活动怎么参加", "coupon_promo"),
        ("我要投诉，转人工", "complaint_escalation"),
        ("你们虚假发货，我要举报", "complaint_escalation"),
    ])
    def test_intent_recognized(self, text, intent):
        decision = route(text)
        assert decision.intent == intent, f"{text} → {decision.intent}（期望 {intent}）"
        assert decision.route == "workflow"
        assert decision.workflow in workflow_names()

    def test_confidence_above_threshold_for_clear_cases(self):
        decision = route("我要退款，订单号 SO2026000001")
        assert decision.confidence >= 0.62
        assert decision.source == "rule"


class TestLongTail:
    @pytest.mark.parametrize("text", [
        "帮我分析一下最近各品类的订单占比",
        "帮我预测下个月的退款趋势",
        "帮我写一段挽回客户的推荐话术",
    ])
    def test_ambiguous_goes_to_agent(self, text):
        decision = route(text)
        assert decision.route == "agent"
        assert decision.workflow == ""

    def test_low_confidence_records_reason(self):
        decision = route("随便问点什么")
        assert decision.route == "agent"
        assert "阈值" in decision.reason or "无确定性工作流" in decision.reason


class TestDisambiguation:
    def test_refund_beats_logistics_on_not_shipped(self):
        """「还没发货，我不要了，退款」不能因为"发货"二字被误判成物流查询。"""
        decision = route("订单 SO2026000002 还没发货，我不要了，退款")
        assert decision.intent == "refund_request"

    def test_negation_does_not_trigger_refund(self):
        decision = route("我不要退款，我只想问订单 SO2026000001 的物流到哪了")
        assert decision.intent == "logistics_query"

    def test_analysis_phrasing_goes_to_agent(self):
        decision = route("帮我分析物流异常率最高的承运商")
        assert decision.route == "agent"

    def test_multiple_orders_is_compound(self):
        decision = route("订单 SO2026000001 和 SO2026000002 都要退款")
        assert decision.route == "agent"
        assert "复合" in decision.reason

    def test_price_mentioned_amount_becomes_current_price(self):
        """价保场景里用户说的"现在卖 343 元"是现价，不是退款金额。"""
        decision = route("订单 SO2026000001 刚买就降价了，现在卖 343 元，能补差价吗")
        assert decision.intent == "price_protection"
        assert decision.slots.get("current_price") == 343.0


class TestPendingIntentCarryover:
    def test_followup_keeps_original_intent(self):
        """追问槽位后的下一句（只回了订单号）必须沿用原意图，否则多轮对话就断了。"""
        decision = route("订单号是 SO2026000124",
                         memory_slots={"_pending_intent": "refund_request"})
        assert decision.intent == "refund_request"
        assert decision.route == "workflow"
        assert decision.source == "pending_intent"

    def test_new_clear_intent_overrides_pending(self):
        decision = route("我要开发票，订单号 SO2026000124",
                         memory_slots={"_pending_intent": "refund_request"})
        assert decision.intent == "invoice_issue"

    def test_pending_ignored_when_no_pending(self):
        decision = route("订单号是 SO2026000124")
        assert decision.route == "agent"      # 没有上下文时无法判断意图


class TestSlotExtraction:
    def test_order_and_customer(self):
        slots = extract_slots("客户 C00001 的订单 SO2026000124")
        assert slots["order_id"] == "SO2026000124"
        assert slots["customer_id"] == "C00001"

    def test_tracking_and_amount(self):
        slots = extract_slots("运单 SF123456789012 退了 199 元")
        assert slots["tracking_no"] == "SF123456789012"
        assert slots["amount"] == 199.0

    def test_current_price_forms(self):
        assert extract_slots("现在卖 343.2")["current_price"] == 343.2
        assert extract_slots("现价 99 元")["current_price"] == 99.0
        assert extract_slots("降到 88")["current_price"] == 88.0

    def test_empty_when_nothing(self):
        assert extract_slots("你好") == {}


class TestRouterStats:
    def test_stats_counted(self):
        router = IntentRouter(llm=None)
        router.route("我要退款，订单号 SO2026000001")
        router.route("帮我分析一下数据占比")
        assert router.stats["rule"] >= 1
        assert router.stats["agent"] >= 1

    def test_routing_table_covers_eight_intents(self):
        assert len(INTENTS) == 8
        table = describe_routing()
        for spec in INTENTS:
            assert spec.label in table
