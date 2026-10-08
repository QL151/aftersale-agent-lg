"""造数：把"每天 800+ 售后工单"的业务现场在本地复现出来。

为什么要认真造数？
  因为简历上的数字（完成率、全自动闭环率、人工介入下降）**必须能被别人在你机器上复现**。
  造数脚本是这些数字的地基：没有它，指标就是编的；有它，面试官可以自己跑一遍。

时间基准取自 app.clock.now()，所以配合 BIZ_NOW 使用即可得到**完全确定性**的数据集。
"""

from __future__ import annotations

import random
import sqlite3

from . import clock
from .config import settings
from .db import Repo, connect, init_schema, is_seeded

# ---------------------------------------------------------------- 类目与售后规则
# clause = 对外可引用的条款号，Agent 回答里必须带上它 —— 这是"可解释"的落地。
RULES: list[tuple] = [
    # category, clause, 无理由天数, 允许拆封, 质量问题天数, 运费可退, 说明
    ("服饰", "售后规则 3.1", 7, 1, 30, 0, "吊牌完整可退，穿过/洗过不支持"),
    ("数码", "售后规则 3.2", 7, 0, 30, 0, "拆封激活后不支持无理由，质量问题走检测"),
    ("美妆", "售后规则 3.3", 7, 0, 30, 0, "拆封后不支持无理由退货"),
    ("生鲜", "售后规则 3.4", 0, 0, 2, 1, "不支持无理由，质量问题 48 小时内受理"),
    ("定制", "售后规则 3.5", 0, 0, 30, 0, "定制品不支持无理由退货"),
    ("图书", "售后规则 3.6", 7, 1, 30, 0, "不影响二次销售可退"),
    ("家电", "售后规则 3.7", 7, 0, 30, 1, "质量问题运费商家承担，需上门检测单"),
]

SKUS: dict[str, list[tuple[str, float]]] = {
    "服饰": [("纯棉圆领T恤", 99.0), ("轻薄羽绒服", 599.0), ("牛仔裤", 259.0), ("运动鞋", 429.0)],
    "数码": [("蓝牙耳机 Pro", 899.0), ("机械键盘", 469.0), ("移动电源 20000mAh", 159.0), ("智能手表", 1299.0)],
    "美妆": [("氨基酸洁面乳", 89.0), ("精华液 30ml", 329.0), ("防晒霜 SPF50", 149.0)],
    "生鲜": [("阳澄湖大闸蟹 4 只", 268.0), ("智利车厘子 2斤", 129.0), ("挪威三文鱼 500g", 158.0)],
    "定制": [("定制刻字项链", 399.0), ("定制抱枕", 89.0), ("定制 T 恤印花", 129.0)],
    "图书": [("《深入理解计算机系统》", 139.0), ("《设计数据密集型应用》", 128.0)],
    "家电": [("变频空调 1.5匹", 2699.0), ("扫地机器人", 1899.0), ("空气炸锅", 399.0)],
}

SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
GIVEN = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "艳", "勇", "军", "杰", "娟", "涛", "明", "超", "秀英", "霞", "平", "刚"]

CARRIERS = ["顺丰速运", "中通快递", "圆通速递", "京东物流", "韵达快递"]

# 工单意图分布 —— 前 8 类是"高频意图"（走确定性工作流），后面是长尾（走自主 Agent）
INTENT_MIX: list[tuple[str, str, int]] = [
    ("logistics_query", "物流查询", 22),
    ("refund_progress", "退款进度查询", 14),
    ("refund_request", "退款申请", 16),
    ("return_exchange", "退货换货", 10),
    ("invoice_issue", "发票问题", 8),
    ("price_protection", "价保补差", 7),
    ("coupon_promo", "优惠券活动咨询", 6),
    ("complaint_escalation", "投诉升级", 5),
    # ---- 长尾 ----
    ("product_consult", "商品咨询", 4),
    ("order_modify", "订单改址/改规格", 3),
    ("account_issue", "账号与登录问题", 2),
    ("other", "其它", 3),
]

KB_ARTICLES: list[tuple[str, str, str, str]] = [
    ("KB001", "七天无理由退货规则", "退货政策",
     "自签收次日起 7 个自然日内，商品不影响二次销售可申请无理由退货。服饰类吊牌需完整；"
     "数码、美妆类拆封后不支持无理由退货，仅支持质量问题检测。定制品、生鲜不支持无理由退货。"),
    ("KB002", "退款到账时效说明", "退款政策",
     "退款审核通过后：原路退回，微信/支付宝 1-3 个工作日到账，银行卡 3-7 个工作日到账。"
     "若超过 7 个工作日未到账，请提供订单号由客服发起支付渠道查询。"),
    ("KB003", "运费退还规则", "退款政策",
     "非商家责任（如七天无理由）退货运费由买家承担，已支付的发货运费不予退还。"
     "商家责任（质量问题、发错货、少件）往返运费由商家承担。生鲜、家电类质量问题运费由商家承担。"),
    ("KB004", "发票开具与换开", "发票",
     "订单完成后 30 天内可申请电子发票，支持个人抬头与企业抬头。企业抬头需提供公司名称与税号。"
     "发票换开需先作废原发票，处理时效 1-3 个工作日。"),
    ("KB005", "价格保护规则", "价保",
     "自签收之日起 7 天内，同一商品在同一渠道官方降价，可申请补差价。"
     "秒杀、拼团、限时优惠、优惠券导致的价差不参与价保。每个订单仅可申请一次价保。"),
    ("KB006", "优惠券使用与补偿", "优惠券",
     "优惠券不可提现、不可转赠，逾期作废。因平台或商家原因导致订单取消的，"
     "可申请等额补偿券，补偿券自发放起 30 天内有效。"),
    ("KB007", "投诉升级与赔付标准", "投诉处理",
     "普通投诉 24 小时内响应。涉及发错货、少件、虚假发货等严重问题，升级至人工专员处理，"
     "赔付标准：少件按缺失商品实付金额赔付；虚假发货赔付订单实付金额的 30%。"),
    ("KB008", "物流异常处理流程", "物流",
     "物流超过 72 小时无更新视为异常。客服需先向承运商发起查询，48 小时内未恢复则按丢件处理，"
     "可为客户补发或全额退款，运费由承运商责任方向商家赔付。"),
    ("KB009", "生鲜类售后特殊规则", "退货政策",
     "生鲜商品不支持七天无理由退货。质量问题（变质、腐坏、死亡）需在签收后 48 小时内"
     "提供照片凭证，核实后按坏损比例退款，无需寄回。"),
    ("KB010", "订单修改与取消", "订单",
     "订单未发货前可申请取消，全额退款；已发货订单需走拒收或退货流程。"
     "收货地址修改仅在未发货状态下支持，且仅限同城范围。"),
    ("KB011", "退货换货操作指引", "退货政策",
     "换货仅支持同款同规格的颜色/尺码更换，且需在签收后 7 天内申请。"
     "换货需先寄回原商品，商家签收后 48 小时内发出新商品。无库存时可转为退款。"),
    ("KB012", "账号安全与登录问题", "账号",
     "账号无法登录可通过绑定手机号验证码找回。若手机号已停用，需提交身份证明由人工审核，"
     "审核时效 1 个工作日。"),
]


def _phone(rng: random.Random) -> str:
    return "1" + rng.choice("35789") + "".join(rng.choice("0123456789") for _ in range(9))


def _name(rng: random.Random) -> str:
    return rng.choice(SURNAMES) + rng.choice(GIVEN)


def _intent_of(rng: random.Random) -> tuple[str, str]:
    pool = [item for item in INTENT_MIX for _ in range(item[2])]
    key, label, _w = rng.choice(pool)
    return key, label


def seed(
    customers: int = 300,
    orders: int = 900,
    tickets: int = 900,
    days_span: int = 30,
    force: bool = False,
) -> dict:
    """幂等地把演示数据灌进 SQLite。返回各表行数。"""
    settings.ensure_dirs()
    conn = connect(settings.db_path)
    try:
        init_schema(conn)
        if is_seeded(conn) and not force:
            return _counts(conn)

        if force:
            for table in (
                "customers", "orders", "shipments", "shipment_events", "refund_rules",
                "tickets", "ticket_notes", "refunds", "coupons", "coupons_issued",
                "kb_articles", "approvals", "events", "meta",
            ):
                conn.execute(f"DELETE FROM {table}")

        rng = random.Random(20250924)  # 固定种子 → 数据可复现

        # ---- 售后规则 ----
        conn.executemany(
            "INSERT OR REPLACE INTO refund_rules (category, clause, max_days_after_delivery,"
            " allow_opened, quality_days, shipping_refundable, note) VALUES (?,?,?,?,?,?,?)",
            RULES,
        )
        # ---- 知识库 ----
        conn.executemany(
            "INSERT OR REPLACE INTO kb_articles (id, title, category, content) VALUES (?,?,?,?)",
            KB_ARTICLES,
        )

        # ---- 客户 ----
        customer_rows = []
        customer_ids = []
        for i in range(customers):
            cid = f"C{i + 1:05d}"
            customer_ids.append(cid)
            level = rng.choices(["普通", "银牌", "金牌", "黑名单观察"], weights=[70, 18, 10, 2])[0]
            refunds_90 = rng.choices([0, 1, 2, 3, 5], weights=[55, 25, 12, 5, 3])[0]
            complaints = rng.choices([0, 1, 2], weights=[85, 12, 3])[0]
            risk = min(0.95, refunds_90 * 0.12 + complaints * 0.2 + (0.3 if level == "黑名单观察" else 0))
            customer_rows.append(
                (cid, _name(rng), _phone(rng), level,
                 clock.ago(rng.uniform(30, 1200)), refunds_90, complaints, round(risk, 3))
            )
        conn.executemany(
            "INSERT INTO customers (customer_id, name, phone, level, registered_at,"
            " refund_count_90d, complaint_count_90d, risk_score) VALUES (?,?,?,?,?,?,?,?)",
            customer_rows,
        )

        # ---- 订单 + 物流 ----
        order_rows, shipment_rows, event_rows = [], [], []
        order_index: list[tuple[str, str, str, float]] = []   # (order_id, cust_id, category, amount)
        for i in range(orders):
            oid = f"SO{clock.now().year}{i + 1:06d}"
            cid = rng.choice(customer_ids)
            category = rng.choices(
                list(SKUS.keys()), weights=[26, 18, 14, 9, 6, 8, 19]
            )[0]
            sku, base = rng.choice(SKUS[category])
            qty = rng.choices([1, 2, 3], weights=[80, 15, 5])[0]
            amount = round(base * qty * rng.choice([1.0, 1.0, 1.0, 0.9, 0.85]), 2)
            shipping = rng.choice([0.0, 0.0, 8.0, 12.0])
            paid_days = rng.uniform(0.2, days_span)
            paid_at = clock.ago(paid_days)
            is_opened = 1 if rng.random() < 0.35 else 0
            promo = round(amount * rng.choice([0.0, 0.0, 0.0, 0.05, 0.1]), 2)

            # 订单状态与签收时间
            roll = rng.random()
            if roll < 0.06:
                status, delivered, shipped = "已取消", "", False
            elif roll < 0.20:
                status, delivered, shipped = "已支付", "", False
            elif roll < 0.55:
                status, delivered, shipped = "已发货", "", True
            elif roll < 0.94:
                delivered_days = max(paid_days - rng.uniform(1, 4), 0.1)
                status, delivered, shipped = "已签收", clock.ago(delivered_days), True
            else:
                delivered_days = max(paid_days - rng.uniform(1, 4), 0.1)
                status, delivered, shipped = "已退款", clock.ago(delivered_days), True

            invoice = rng.choices(["未开票", "已开票", "开票中"], weights=[70, 25, 5])[0]
            order_rows.append(
                (oid, cid, sku, category, amount, shipping, paid_at, delivered, status,
                 invoice, 7, is_opened, promo)
            )
            order_index.append((oid, cid, category, amount))

            if shipped:
                tracking = f"SF{rng.randint(10**11, 10**12 - 1)}"
                carrier = rng.choice(CARRIERS)
                if status == "已签收" or status == "已退款":
                    ship_status = "已签收"
                elif rng.random() < 0.08:
                    ship_status = "异常"
                else:
                    ship_status = rng.choices(["已揽收", "运输中", "派送中"], weights=[15, 55, 30])[0]
                shipment_rows.append((tracking, oid, carrier, ship_status))

                base_time = paid_days - 0.2
                steps = [
                    ("已揽收", "杭州市余杭区", "快件已由揽收员揽收"),
                    ("运输中", "杭州转运中心", "快件已到达杭州转运中心"),
                    ("运输中", "上海分拨中心", "快件已发往上海分拨中心"),
                    ("派送中", "上海市浦东新区", "快件正在派送中，请保持电话畅通"),
                ]
                if ship_status == "异常":
                    steps = [
                        ("已揽收", "杭州市余杭区", "快件已由揽收员揽收"),
                        ("运输中", "杭州转运中心", "快件已到达杭州转运中心"),
                        ("异常", "未知", "物流信息超过 72 小时未更新，已发起承运商查询"),
                    ]
                if ship_status == "已签收":
                    steps.append(("已签收", "上海市浦东新区", "快件已签收，签收人：本人"))
                for k, (st, loc, desc) in enumerate(steps):
                    event_rows.append(
                        (tracking, clock.ago(max(base_time - k * 0.25, 0.05)), st, loc, desc)
                    )

        conn.executemany(
            "INSERT INTO orders (order_id, customer_id, sku_name, category, amount, shipping_fee,"
            " paid_at, delivered_at, status, invoice_status, price_protect_days, is_opened,"
            " promo_deduct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            order_rows,
        )
        conn.executemany(
            "INSERT INTO shipments (tracking_no, order_id, carrier, status) VALUES (?,?,?,?)",
            shipment_rows,
        )
        conn.executemany(
            "INSERT INTO shipment_events (tracking_no, time, status, location, description)"
            " VALUES (?,?,?,?,?)",
            event_rows,
        )

        # ---- 历史工单 ----
        # 造的是"改造前的现场"：绝大多数靠人工，且每条都记录了人工耗时（基线 6 分钟）。
        ticket_rows = []
        for i in range(tickets):
            oid, cid, _cat, _amt = rng.choice(order_index)
            intent, label = _intent_of(rng)
            created_days = rng.uniform(0, days_span)
            manual = rng.random() < 0.95
            assignee = "人工坐席" if manual else "AI"
            auto = 0 if manual else 1
            status = rng.choices(["已关闭", "已关闭", "已关闭", "处理中", "待人工"], weights=[50, 20, 15, 10, 5])[0]
            base_min = round(rng.gauss(6.0, 2.0), 1)
            base_min = max(1.5, min(base_min, 15.0))
            ticket_rows.append(
                (f"T{clock.now().strftime('%y%m')}{i + 1:05d}", oid, cid, intent,
                 f"{label}：订单 {oid}", status, assignee, clock.ago(created_days),
                 clock.ago(max(created_days - 0.05, 0)), clock.ago(max(created_days - 0.1, 0)) if status == "已关闭" else "",
                 auto, rng.choice(["在线客服", "电话", "APP", "小程序"]), base_min,
                 round(base_min if not auto else 0.2, 1))
            )
        conn.executemany(
            "INSERT INTO tickets (ticket_id, order_id, customer_id, intent, title, status,"
            " assignee, created_at, updated_at, closed_at, auto_closed, channel,"
            " baseline_minutes, handled_minutes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ticket_rows,
        )

        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('seeded_at', ?)", (clock.stamp(),)
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('seed_span_days', ?)", (str(days_span),)
        )
        conn.commit()
        return _counts(conn)
    finally:
        conn.close()


def _counts(conn: sqlite3.Connection) -> dict:
    tables = (
        "customers", "orders", "shipments", "shipment_events", "refund_rules",
        "tickets", "kb_articles",
    )
    out = {}
    for t in tables:
        out[t] = conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
    return out


def ensure_seeded() -> dict:
    """第一次用到时自动造数 —— 让"clone 下来就能跑"成立。"""
    settings.ensure_dirs()
    conn = connect(settings.db_path)
    try:
        init_schema(conn)
        if is_seeded(conn):
            return _counts(conn)
    finally:
        conn.close()
    return seed()


if __name__ == "__main__":  # python -m app.seed [--force]
    import sys

    result = seed(force="--force" in sys.argv)
    print("售后演示数据就绪：")
    for key, value in result.items():
        print(f"  {key:<18}{value}")
    with Repo() as repo:
        stats = repo.ticket_stats()
    print(f"  历史工单自动闭环率 {stats['auto_rate'] * 100:.1f}%（改造前基线）")
