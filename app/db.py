"""业务数据层：订单 / 物流 / 客户 / 工单 / 退款 / 优惠券 / 知识库 / 审批单。

生产里这些分散在订单系统、物流网关、CRM、工单系统里，由 MCP Server 统一封装（见 mcp_server/）；
本项目用一张 SQLite 把它们落在本地，好处是**零依赖可跑、可复现、可评测**。

两个面试点写在代码里：
  1. **幂等**：refunds 表对 idempotency_key 建唯一索引。
     退款是"绝对不能重复扣款"的动作，重试必须无害 —— 这是业务 Agent 与玩具 Agent 的分水岭。
  2. **审计**：approvals 表记录每一次人工审批的 who/when/what/why，只增不改。
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from . import clock
from .config import settings
from .schemas import (
    Customer,
    Order,
    PolicyVerdict,
    RefundQuote,
    Shipment,
    ShipmentEvent,
    Ticket,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY, name TEXT NOT NULL, phone TEXT NOT NULL,
    level TEXT NOT NULL, registered_at TEXT NOT NULL,
    refund_count_90d INTEGER NOT NULL DEFAULT 0,
    complaint_count_90d INTEGER NOT NULL DEFAULT 0,
    risk_score REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, sku_name TEXT NOT NULL,
    category TEXT NOT NULL, amount REAL NOT NULL, shipping_fee REAL NOT NULL DEFAULT 0,
    paid_at TEXT NOT NULL, delivered_at TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
    invoice_status TEXT NOT NULL DEFAULT '未开票', price_protect_days INTEGER NOT NULL DEFAULT 7,
    is_opened INTEGER NOT NULL DEFAULT 0, promo_deduct REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS shipments (
    tracking_no TEXT PRIMARY KEY, order_id TEXT NOT NULL, carrier TEXT NOT NULL,
    status TEXT NOT NULL, abnormal_reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS shipment_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tracking_no TEXT NOT NULL,
    time TEXT NOT NULL, status TEXT NOT NULL, location TEXT NOT NULL, description TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS refund_rules (
    category TEXT PRIMARY KEY, clause TEXT NOT NULL, max_days_after_delivery INTEGER NOT NULL,
    allow_opened INTEGER NOT NULL, quality_days INTEGER NOT NULL DEFAULT 30,
    shipping_refundable INTEGER NOT NULL DEFAULT 0, note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, customer_id TEXT NOT NULL,
    intent TEXT NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL, assignee TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, closed_at TEXT NOT NULL DEFAULT '',
    auto_closed INTEGER NOT NULL DEFAULT 0, channel TEXT NOT NULL DEFAULT '在线客服',
    baseline_minutes REAL NOT NULL DEFAULT 6.0, handled_minutes REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ticket_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT NOT NULL, author TEXT NOT NULL,
    content TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS refunds (
    refund_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, amount REAL NOT NULL,
    reason TEXT NOT NULL, operator TEXT NOT NULL, created_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS coupons (
    coupon_id TEXT PRIMARY KEY, code TEXT NOT NULL, customer_id TEXT NOT NULL,
    amount REAL NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
    expire_at TEXT NOT NULL, reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS coupons_issued (
    id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT NOT NULL, code TEXT NOT NULL,
    amount REAL NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS kb_articles (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, category TEXT NOT NULL, content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, thread_id TEXT NOT NULL,
    action TEXT NOT NULL, arguments TEXT NOT NULL, reason TEXT NOT NULL,
    risk_level TEXT NOT NULL, amount REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL, created_at TEXT NOT NULL, decided_at TEXT NOT NULL DEFAULT '',
    decided_by TEXT NOT NULL DEFAULT '', decision_note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, kind TEXT NOT NULL,
    payload TEXT NOT NULL, created_at TEXT NOT NULL
);
-- 长期用户偏好：生产里在 PostgreSQL（见 app/memory/profile.py），这里是无 PG 时的降级表。
-- 表结构刻意与 PG 版本保持一致（复合主键 + source + updated_at），
-- 这样切换存储时上层的 SQL 语义不用变。
CREATE TABLE IF NOT EXISTS customer_preferences (
    customer_id TEXT NOT NULL, pref_key TEXT NOT NULL, pref_value TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'agent', updated_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, pref_key)
);
CREATE TABLE IF NOT EXISTS customer_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT NOT NULL, fact TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
"""


class _LockedConnection:
    """给 sqlite3 连接套一把锁。

    为什么需要？
      工具执行走线程池（为了做超时保护），于是同一个连接会被**不同线程**使用。
      sqlite3 默认 check_same_thread=True 会直接抛错；
      就算关掉它，也需要自己保证不并发使用同一连接。
      这里用一把 RLock 串行化所有语句，是"本地单机 + 线程池"场景下最省事且正确的做法。
      （生产上换 PostgreSQL + 连接池就没这个问题 —— 这也是本项目把 Repo 抽象出来的原因。）
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()

    def execute(self, *args, **kwargs):
        with self._lock:
            return self._conn.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        with self._lock:
            return self._conn.executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        with self._lock:
            return self._conn.executescript(*args, **kwargs)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __getattr__(self, item):
        return getattr(self._conn, item)


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or settings.db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")     # 允许读写并发，减少 "database is locked"
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def is_seeded(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'seeded_at'").fetchone()
    except sqlite3.OperationalError:
        return False
    return row is not None


def reset_db(path: Path | None = None) -> None:
    target = path or settings.db_path
    if target.exists():
        target.unlink()
    conn = connect(target)
    try:
        init_schema(conn)
    finally:
        conn.close()


# ==================================================================== 仓储


class Repo:
    """所有业务读写的唯一入口。工具层只调它，不直接碰 SQL。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.db_path
        self._conn = _LockedConnection(connect(self.path))
        init_schema(self._conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Repo":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---------------------------------------------------------------- 客户
    def get_customer(self, customer_id: str) -> Customer | None:
        row = self._conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        return _customer(row) if row else None

    def find_customers_by_name(self, name: str) -> list[Customer]:
        rows = self._conn.execute(
            "SELECT * FROM customers WHERE name LIKE ? LIMIT 20", (f"%{name}%",)
        ).fetchall()
        return [_customer(r) for r in rows]

    # ---------------------------------------------------------------- 订单
    def get_order(self, order_id: str) -> Order | None:
        row = self._conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        return self._order(row) if row else None

    def list_orders_by_customer(self, customer_id: str, limit: int = 10) -> list[Order]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE customer_id = ? ORDER BY paid_at DESC LIMIT ?",
            (customer_id, limit),
        ).fetchall()
        return [self._order(r) for r in rows]

    def _order(self, row: sqlite3.Row) -> Order:
        customer = self.get_customer(row["customer_id"])
        return Order(
            order_id=row["order_id"],
            customer_id=row["customer_id"],
            customer_name=customer.name if customer else "未知",
            contact=customer.phone if customer else "",
            sku_name=row["sku_name"],
            category=row["category"],
            amount=row["amount"],
            shipping_fee=row["shipping_fee"],
            paid_at=row["paid_at"],
            delivered_at=row["delivered_at"],
            status=row["status"],
            invoice_status=row["invoice_status"],
            price_protect_days=row["price_protect_days"],
            is_opened=row["is_opened"],
        )

    def promo_deduct(self, order_id: str) -> float:
        row = self._conn.execute(
            "SELECT promo_deduct FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        return float(row["promo_deduct"]) if row else 0.0

    def update_order_status(self, order_id: str, status: str) -> None:
        self._conn.execute("UPDATE orders SET status = ? WHERE order_id = ?", (status, order_id))
        self._conn.commit()

    def set_invoice_status(self, order_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE orders SET invoice_status = ? WHERE order_id = ?", (status, order_id)
        )
        self._conn.commit()

    # ---------------------------------------------------------------- 物流
    def get_shipment_by_order(self, order_id: str) -> Shipment | None:
        row = self._conn.execute(
            "SELECT * FROM shipments WHERE order_id = ? LIMIT 1", (order_id,)
        ).fetchone()
        if not row:
            return None
        events = self._conn.execute(
            "SELECT * FROM shipment_events WHERE tracking_no = ? ORDER BY id", (row["tracking_no"],)
        ).fetchall()
        return Shipment(
            tracking_no=row["tracking_no"],
            carrier=row["carrier"],
            order_id=row["order_id"],
            status=row["status"],
            events=[
                ShipmentEvent(
                    time=e["time"], status=e["status"], location=e["location"],
                    description=e["description"],
                )
                for e in events
            ],
        )

    def get_shipment_by_tracking(self, tracking_no: str) -> Shipment | None:
        row = self._conn.execute(
            "SELECT order_id FROM shipments WHERE tracking_no = ?", (tracking_no,)
        ).fetchone()
        return self.get_shipment_by_order(row["order_id"]) if row else None

    def abnormal_shipments(self, limit: int = 5) -> list[Shipment]:
        rows = self._conn.execute(
            "SELECT order_id FROM shipments WHERE status = '异常' LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            s = self.get_shipment_by_order(r["order_id"])
            if s:
                out.append(s)
        return out

    # ---------------------------------------------------------------- 规则
    def get_rule(self, category: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM refund_rules WHERE category = ?", (category,)
        ).fetchone()
        return dict(row) if row else None

    def all_rules(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM refund_rules").fetchall()]

    # ---------------------------------------------------------------- 工单
    def create_ticket(
        self, order_id: str, customer_id: str, intent: str, title: str, channel: str = "在线客服"
    ) -> Ticket:
        now = clock.stamp()
        ticket_id = f"T{clock.now().strftime('%y%m%d')}{uuid.uuid4().hex[:6].upper()}"
        self._conn.execute(
            "INSERT INTO tickets (ticket_id, order_id, customer_id, intent, title, status,"
            " assignee, created_at, updated_at, channel) VALUES (?,?,?,?,?,'处理中','AI',?,?,?)",
            (ticket_id, order_id, customer_id, intent, title, now, now, channel),
        )
        self._conn.commit()
        return self.get_ticket(ticket_id)  # type: ignore[return-value]

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        row = self._conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        return _ticket(row) if row else None

    def find_tickets(self, order_id: str = "", status: str = "", limit: int = 20) -> list[Ticket]:
        sql = "SELECT * FROM tickets WHERE 1=1"
        params: list[Any] = []
        if order_id:
            sql += " AND order_id = ?"
            params.append(order_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [_ticket(r) for r in self._conn.execute(sql, params).fetchall()]

    def add_note(self, ticket_id: str, author: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO ticket_notes (ticket_id, author, content, created_at) VALUES (?,?,?,?)",
            (ticket_id, author, content, clock.stamp()),
        )
        self._conn.execute(
            "UPDATE tickets SET updated_at = ? WHERE ticket_id = ?", (clock.stamp(), ticket_id)
        )
        self._conn.commit()

    def update_ticket(self, ticket_id: str, **fields: Any) -> None:
        allowed = {"status", "assignee", "closed_at", "auto_closed"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        sets["updated_at"] = clock.stamp()
        clause = ", ".join(f"{k} = ?" for k in sets)
        self._conn.execute(
            f"UPDATE tickets SET {clause} WHERE ticket_id = ?", [*sets.values(), ticket_id]
        )
        self._conn.commit()

    # ---------------------------------------------------------------- 退款
    def find_refund(self, idempotency_key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM refunds WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return dict(row) if row else None

    def insert_refund(
        self, order_id: str, amount: float, reason: str, operator: str, idempotency_key: str
    ) -> dict[str, Any]:
        refund_id = f"R{uuid.uuid4().hex[:10].upper()}"
        self._conn.execute(
            "INSERT INTO refunds (refund_id, order_id, amount, reason, operator, created_at,"
            " idempotency_key) VALUES (?,?,?,?,?,?,?)",
            (refund_id, order_id, amount, reason, operator, clock.stamp(), idempotency_key),
        )
        self._conn.commit()
        return {"refund_id": refund_id, "order_id": order_id, "amount": amount}

    # ---------------------------------------------------------------- 优惠券
    def grant_coupon(
        self, customer_id: str, amount: float, reason: str, idempotency_key: str, days: int = 30
    ) -> dict[str, Any]:
        code = f"CPN{abs(hash(idempotency_key)) % 10**8:08d}"
        self._conn.execute(
            "INSERT INTO coupons_issued (customer_id, code, amount, reason, created_at,"
            " idempotency_key) VALUES (?,?,?,?,?,?)",
            (customer_id, code, amount, reason, clock.stamp(), idempotency_key),
        )
        self._conn.execute(
            "INSERT INTO coupons (coupon_id, code, customer_id, amount, status, created_at,"
            " expire_at, reason) VALUES (?,?,?,?,'未使用',?,?,?)",
            (
                f"CP{uuid.uuid4().hex[:8].upper()}", code, customer_id, amount,
                clock.stamp(), clock.ahead(days), reason,
            ),
        )
        self._conn.commit()
        return {"code": code, "amount": amount, "customer_id": customer_id}

    def list_coupons(self, customer_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM coupons WHERE customer_id = ? ORDER BY created_at DESC LIMIT 10",
                (customer_id,),
            ).fetchall()
        ]

    # ---------------------------------------------------------------- 知识库
    def search_kb(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """朴素关键词检索。

        生产里这里是 RAG（向量检索 + 重排）；本项目刻意用关键词打分，
        好处是**离线可测**，且能保证评测集里"必须命中某条款"的断言是稳定的。
        接 RAG 只需替换这个方法 —— 工具层和 Agent 层完全不用动。
        """
        tokens = [t for t in _tokenize(query) if len(t) >= 2]
        rows = self._conn.execute("SELECT * FROM kb_articles").fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            article = dict(row)
            haystack = f"{article['title']} {article['category']} {article['content']}"
            score = sum(haystack.count(t) for t in tokens) / max(len(tokens), 1)
            if score > 0:
                scored.append((score, article))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:  # 兜底：返回最相关的制度类文章，避免"检索为空"让 Agent 卡住
            return [dict(r) for r in rows[:limit]]
        return [a for _s, a in scored[:limit]]

    # ---------------------------------------------------------------- 审批单
    def create_approval(
        self, approval_id: str, run_id: str, thread_id: str, action: str,
        arguments: str, reason: str, risk_level: str, amount: float,
    ) -> None:
        # INSERT OR IGNORE + 确定性 approval_id：
        # 图被 resume 时节点会**从头重跑**，会再调一次本方法，必须幂等。
        self._conn.execute(
            "INSERT OR IGNORE INTO approvals (approval_id, run_id, thread_id, action, arguments,"
            " reason, risk_level, amount, status, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,'pending',?)",
            (approval_id, run_id, thread_id, action, arguments, reason, risk_level, amount,
             clock.stamp()),
        )
        self._conn.commit()

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_approvals(self, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM approvals WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM approvals ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def decide_approval(
        self, approval_id: str, status: str, decided_by: str, note: str = ""
    ) -> bool:
        cur = self._conn.execute(
            "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ?, decision_note = ?"
            " WHERE approval_id = ? AND status = 'pending'",
            (status, clock.stamp(), decided_by, note, approval_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ---------------------------------------------------------------- 事件
    def log_event(self, run_id: str, kind: str, payload: str) -> None:
        self._conn.execute(
            "INSERT INTO events (run_id, kind, payload, created_at) VALUES (?,?,?,?)",
            (run_id, kind, payload, clock.stamp()),
        )
        self._conn.commit()

    # ---------------------------------------------------------------- 通用查询
    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """给分析类工具用的只读查询入口（SQL 由本模块内部拼好，不接受模型输入）。"""
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # ---------------------------------------------------------------- 指标
    def ticket_stats(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) c FROM tickets").fetchone()["c"]
        auto = self._conn.execute(
            "SELECT COUNT(*) c FROM tickets WHERE auto_closed = 1"
        ).fetchone()["c"]
        manual = self._conn.execute(
            "SELECT COUNT(*) c FROM tickets WHERE assignee != 'AI'"
        ).fetchone()["c"]
        pending = self._conn.execute(
            "SELECT COUNT(*) c FROM tickets WHERE status = '待人工'"
        ).fetchone()["c"]
        return {
            "tickets_total": total,
            "auto_closed": auto,
            "manual": manual,
            "pending_manual": pending,
            "auto_rate": round(auto / total, 4) if total else 0.0,
        }


def _customer(row: sqlite3.Row) -> Customer:
    return Customer(
        customer_id=row["customer_id"], name=row["name"], phone=row["phone"],
        level=row["level"], registered_at=row["registered_at"],
        refund_count_90d=row["refund_count_90d"],
        complaint_count_90d=row["complaint_count_90d"], risk_score=row["risk_score"],
    )


def _ticket(row: sqlite3.Row) -> Ticket:
    return Ticket(
        ticket_id=row["ticket_id"], order_id=row["order_id"], customer_id=row["customer_id"],
        intent=row["intent"], title=row["title"], status=row["status"],
        assignee=row["assignee"], created_at=row["created_at"], updated_at=row["updated_at"],
        closed_at=row["closed_at"], auto_closed=row["auto_closed"],
    )


def _tokenize(text: str) -> list[str]:
    """中文按 2-gram 粗切 + 英文/数字整词 —— 够用的关键词检索。"""
    import re

    words = re.findall(r"[A-Za-z0-9]+", text)
    chars = re.sub(r"[^\u4e00-\u9fff]", "", text)
    grams = [chars[i : i + 2] for i in range(max(len(chars) - 1, 0))]
    return [*words, *grams]


# ==================================================================== 领域计算
# 下面两个函数是"退款计算"和"规则校验"的纯逻辑实现。
# 工具层（app/tools/refund.py）只是给它们套一层给模型看的描述 —— 业务逻辑不写在工具里。


def check_policy(order: Order, rule: dict[str, Any] | None, quality_issue: bool = False) -> PolicyVerdict:
    """按类目规则 + 时间 + 拆封状态判定能否退。必须给出条款依据。"""
    if rule is None:
        return PolicyVerdict(False, f"类目「{order.category}」没有可用的售后规则", needs_manual=True)

    clause = rule["clause"]
    if order.status in {"已退款"}:
        return PolicyVerdict(False, "该订单已退款，请勿重复申请", clause=clause)
    if order.status == "已取消":
        return PolicyVerdict(False, "订单已取消，无需退款", clause=clause)

    # 未发货：可直接全额退
    if not order.delivered_at:
        return PolicyVerdict(
            True, "订单尚未签收/未发货，可直接申请退款", clause=f"{clause}（未发货场景）",
            max_refund=order.amount + order.shipping_fee,
        )

    elapsed = clock.days_since(order.delivered_at)
    if elapsed is None:
        return PolicyVerdict(False, "签收时间缺失，无法判定时效", needs_manual=True, clause=clause)

    if quality_issue:
        if elapsed <= rule["quality_days"]:
            return PolicyVerdict(
                True, f"质量问题，签收后 {elapsed:.1f} 天（≤{rule['quality_days']} 天）",
                clause=f"{clause} 质量问题条款", max_refund=order.amount + order.shipping_fee,
            )
        return PolicyVerdict(
            False, f"质量问题申诉期为 {rule['quality_days']} 天，已超期 {elapsed:.1f} 天",
            clause=f"{clause} 质量问题条款", needs_manual=True,
        )

    if elapsed > rule["max_days_after_delivery"]:
        return PolicyVerdict(
            False,
            f"已签收 {elapsed:.1f} 天，超过 {rule['max_days_after_delivery']} 天无理由期限",
            clause=f"{clause} 无理由退货期限",
        )

    if order.is_opened and not rule["allow_opened"]:
        return PolicyVerdict(
            False, "商品已拆封，该类目拆封后不支持无理由退货",
            clause=f"{clause} 拆封条款", needs_manual=True,
        )

    return PolicyVerdict(
        True, f"签收后 {elapsed:.1f} 天，在 {rule['max_days_after_delivery']} 天无理由期内",
        clause=f"{clause} 无理由退货条款", max_refund=order.amount,
    )


def calc_refund(
    order: Order, promo_deduct: float, rule: dict[str, Any] | None, quality_issue: bool = False
) -> RefundQuote:
    """退款金额计算：货款 + 运费退还 - 优惠分摊 - 折旧。每一笔都要能解释。"""
    breakdown: list[str] = []
    goods = round(order.amount, 2)
    breakdown.append(f"商品实付：{goods} 元")

    shipping = 0.0
    if rule and rule["shipping_refundable"]:
        shipping = round(order.shipping_fee, 2)
        breakdown.append(f"运费退还：{shipping} 元（该规则运费由商家承担）")
    elif not order.delivered_at:
        shipping = round(order.shipping_fee, 2)
        breakdown.append(f"运费退还：{shipping} 元（未发货，运费应退）")
    else:
        breakdown.append("运费不退：非商家责任")

    deduct = 0.0
    if promo_deduct > 0:
        deduct = round(promo_deduct, 2)
        breakdown.append(f"优惠分摊扣回：-{deduct} 元（满减/券已按比例摊到本单）")

    depreciation = 0.0
    if quality_issue and order.is_opened:
        depreciation = 0.0
        breakdown.append("折旧：0 元（质量问题不折旧）")
    elif order.is_opened and order.category in {"数码", "美妆"}:
        depreciation = round(goods * 0.1, 2)
        breakdown.append(f"折旧：-{depreciation} 元（已拆封数码/美妆按 10% 计）")

    total = round(max(goods + shipping - deduct - depreciation, 0.0), 2)
    breakdown.append(f"应退合计：{total} 元")
    return RefundQuote(
        order_id=order.order_id, goods_amount=goods, shipping_refund=shipping,
        promo_deduct=deduct, depreciation=depreciation, total=total, breakdown=breakdown,
    )
