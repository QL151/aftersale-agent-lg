"""领域逻辑测试：时钟注入、售后政策判定、退款金额计算。

这三块是"业务正确性"的地基：Agent 说错话还能忍，**算错钱不能忍**。
"""

from __future__ import annotations

import os

import pytest

from app import clock
from app.db import calc_refund, check_policy
from app.schemas import Order


def make_order(**kwargs) -> Order:
    base = dict(
        order_id="SO0000000001", customer_id="C00001", customer_name="测试",
        contact="13800000000", sku_name="测试商品", category="服饰",
        amount=100.0, shipping_fee=10.0, paid_at=clock.ago(10),
        delivered_at=clock.ago(3), status="已签收", invoice_status="未开票",
        price_protect_days=7, is_opened=0,
    )
    base.update(kwargs)
    return Order(**base)


RULE_CLOTHES = {
    "category": "服饰", "clause": "售后规则 3.1", "max_days_after_delivery": 7,
    "allow_opened": 1, "quality_days": 30, "shipping_refundable": 0, "note": "",
}
RULE_DIGITAL = {**RULE_CLOTHES, "category": "数码", "clause": "售后规则 3.2",
                "allow_opened": 0}


class TestClock:
    def test_biz_now_override(self, monkeypatch):
        """BIZ_NOW 必须能钉死"业务当前时间"，否则时效规则没法测。"""
        monkeypatch.setenv("BIZ_NOW", "2026-01-10 12:00:00")
        assert clock.now().strftime("%Y-%m-%d") == "2026-01-10"
        assert clock.today_str() == "2026-01-10"

    def test_days_since(self, monkeypatch):
        monkeypatch.setenv("BIZ_NOW", "2026-01-10 12:00:00")
        assert clock.days_since("2026-01-08 12:00:00") == pytest.approx(2.0)
        assert clock.days_since("") is None
        assert clock.days_since("不是时间") is None

    def test_ago_ahead_roundtrip(self, monkeypatch):
        monkeypatch.setenv("BIZ_NOW", "2026-01-10 12:00:00")
        assert clock.days_since(clock.ago(5)) == pytest.approx(5.0, abs=0.01)
        assert clock.days_since(clock.ahead(5)) == pytest.approx(-5.0, abs=0.01)


class TestPolicy:
    def test_within_window_allowed(self):
        verdict = check_policy(make_order(delivered_at=clock.ago(3)), RULE_CLOTHES)
        assert verdict.allowed
        assert "3.1" in verdict.clause          # 必须给出可引用的条款依据

    def test_overdue_denied_with_clause(self):
        verdict = check_policy(make_order(delivered_at=clock.ago(20)), RULE_CLOTHES)
        assert not verdict.allowed
        assert "3.1" in verdict.clause
        assert "超过" in verdict.reason

    def test_opened_digital_denied_but_manual(self):
        """数码拆封后不支持无理由 → 拒绝，但要标记"需人工"（质量问题可走另一条通道）。"""
        verdict = check_policy(
            make_order(category="数码", is_opened=1, delivered_at=clock.ago(2)), RULE_DIGITAL
        )
        assert not verdict.allowed
        assert verdict.needs_manual

    def test_opened_clothes_allowed(self):
        """服饰允许拆封（吊牌完整即可）。"""
        verdict = check_policy(make_order(is_opened=1, delivered_at=clock.ago(2)), RULE_CLOTHES)
        assert verdict.allowed

    def test_quality_issue_gets_longer_window(self):
        verdict = check_policy(
            make_order(category="数码", is_opened=1, delivered_at=clock.ago(20)),
            RULE_DIGITAL, quality_issue=True,
        )
        assert verdict.allowed
        assert "质量问题" in verdict.clause

    def test_not_delivered_always_refundable(self):
        verdict = check_policy(make_order(status="已支付", delivered_at=""), RULE_CLOTHES)
        assert verdict.allowed
        assert verdict.max_refund == 110.0

    def test_already_refunded(self):
        verdict = check_policy(make_order(status="已退款"), RULE_CLOTHES)
        assert not verdict.allowed
        assert "已退款" in verdict.reason

    def test_no_rule_needs_manual(self):
        verdict = check_policy(make_order(category="未知类目"), None)
        assert not verdict.allowed and verdict.needs_manual


class TestRefundCalc:
    def test_basic_no_shipping_refund(self):
        quote = calc_refund(make_order(amount=100, shipping_fee=10), 0.0, RULE_CLOTHES)
        assert quote.total == 100.0
        assert quote.shipping_refund == 0.0
        assert any("运费不退" in b for b in quote.breakdown)

    def test_promo_deduction(self):
        quote = calc_refund(make_order(amount=100), 15.0, RULE_CLOTHES)
        assert quote.total == 85.0

    def test_shipping_refundable_category(self):
        rule = {**RULE_CLOTHES, "shipping_refundable": 1}
        quote = calc_refund(make_order(amount=100, shipping_fee=12), 0.0, rule)
        assert quote.total == 112.0

    def test_depreciation_on_opened_digital(self):
        """拆封数码按 10% 折旧 —— 这是平台和客户最容易吵起来的地方，必须有据可依。"""
        quote = calc_refund(
            make_order(category="数码", amount=1000, is_opened=1), 0.0,
            {**RULE_DIGITAL, "shipping_refundable": 0},
        )
        assert quote.depreciation == 100.0
        assert quote.total == 900.0

    def test_quality_issue_no_depreciation(self):
        quote = calc_refund(
            make_order(category="数码", amount=1000, is_opened=1), 0.0, RULE_DIGITAL,
            quality_issue=True,
        )
        assert quote.depreciation == 0.0
        assert quote.total == 1000.0

    def test_total_never_negative(self):
        quote = calc_refund(make_order(amount=10), 999.0, RULE_CLOTHES)
        assert quote.total == 0.0

    def test_breakdown_is_explainable(self):
        """每一笔扣减都要能说清来源 —— 这是客服答复客户的依据。"""
        quote = calc_refund(make_order(category="数码", amount=1000, is_opened=1),
                            50.0, RULE_DIGITAL)
        assert len(quote.breakdown) >= 4
        assert any("优惠分摊" in b for b in quote.breakdown)
        assert any("折旧" in b for b in quote.breakdown)
        assert quote.total == pytest.approx(850.0)
