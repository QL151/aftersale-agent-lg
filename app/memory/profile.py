"""长期用户偏好（跨会话）：PostgreSQL 为主，SQLite 降级。

短期记忆（session.py）回答"这次对话说到哪了"；
长期偏好回答"**这个客户是谁、他习惯怎么被服务**"：
  · 常用联系方式（短信 / APP 推送）
  · 历次纠纷的结案方式（退款 / 补券 / 换货）
  · 服务敏感点（对时效敏感、要求人工、不接受电话回访）
  · 风险标记（多次退款、需要复核）

这些东西让 Agent 第二次服务同一个客户时**不必再从零问一遍**，
也是"人工介入量下降"的隐形来源：少问一句就少一次转人工。

存储结构在 PG 和 SQLite 上刻意保持一致（复合主键 + source + updated_at），
所以切换存储时上层逻辑一行都不用改。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .. import clock
from ..config import settings
from ..db import connect, init_schema
from ..runtime import get_postgres

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS customer_preferences (
    customer_id TEXT NOT NULL,
    pref_key    TEXT NOT NULL,
    pref_value  TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'agent',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, pref_key)
);
CREATE TABLE IF NOT EXISTS customer_facts (
    id          BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    fact        TEXT NOT NULL,
    run_id      TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_facts_customer ON customer_facts(customer_id, created_at DESC);
"""


# ==================================================================== 实现


class SQLiteProfileStore:
    """降级实现：跟业务库共用 SQLite，零依赖。"""

    name = "sqlite"

    def __init__(self, path: Any = None) -> None:
        self.path = path or settings.db_path
        conn = connect(self.path)
        try:
            init_schema(conn)
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        return connect(self.path)

    def set_preference(self, customer_id: str, key: str, value: Any, source: str = "agent") -> None:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO customer_preferences (customer_id, pref_key, pref_value, source,"
                " updated_at) VALUES (?,?,?,?,?)"
                " ON CONFLICT(customer_id, pref_key) DO UPDATE SET"
                " pref_value = excluded.pref_value, source = excluded.source,"
                " updated_at = excluded.updated_at",
                (customer_id, key, text, source, clock.stamp()),
            )
            conn.commit()
        finally:
            conn.close()

    def all_preferences(self, customer_id: str) -> dict[str, Any]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT pref_key, pref_value FROM customer_preferences WHERE customer_id = ?",
                (customer_id,),
            ).fetchall()
        finally:
            conn.close()
        return {r["pref_key"]: _maybe_json(r["pref_value"]) for r in rows}

    def add_fact(self, customer_id: str, fact: str, run_id: str = "") -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO customer_facts (customer_id, fact, run_id, created_at) VALUES (?,?,?,?)",
                (customer_id, fact, run_id, clock.stamp()),
            )
            conn.commit()
        finally:
            conn.close()

    def recent_facts(self, customer_id: str, limit: int = 5) -> list[str]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT fact FROM customer_facts WHERE customer_id = ? ORDER BY id DESC LIMIT ?",
                (customer_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [r["fact"] for r in rows]


class PostgresProfileStore:
    """生产实现：真正落到 PostgreSQL，跨服务共享。"""

    name = "postgresql"

    def __init__(self, factory: Any) -> None:
        self.factory = factory
        with self.factory() as conn:
            conn.execute(PG_SCHEMA)
            conn.commit()

    def set_preference(self, customer_id: str, key: str, value: Any, source: str = "agent") -> None:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        with self.factory() as conn:
            conn.execute(
                "INSERT INTO customer_preferences (customer_id, pref_key, pref_value, source,"
                " updated_at) VALUES (%s,%s,%s,%s, now())"
                " ON CONFLICT (customer_id, pref_key) DO UPDATE SET"
                " pref_value = EXCLUDED.pref_value, source = EXCLUDED.source,"
                " updated_at = now()",
                (customer_id, key, text, source),
            )
            conn.commit()

    def all_preferences(self, customer_id: str) -> dict[str, Any]:
        with self.factory() as conn:
            rows = conn.execute(
                "SELECT pref_key, pref_value FROM customer_preferences WHERE customer_id = %s",
                (customer_id,),
            ).fetchall()
        return {row[0]: _maybe_json(row[1]) for row in rows}

    def add_fact(self, customer_id: str, fact: str, run_id: str = "") -> None:
        with self.factory() as conn:
            conn.execute(
                "INSERT INTO customer_facts (customer_id, fact, run_id) VALUES (%s,%s,%s)",
                (customer_id, fact, run_id),
            )
            conn.commit()

    def recent_facts(self, customer_id: str, limit: int = 5) -> list[str]:
        with self.factory() as conn:
            rows = conn.execute(
                "SELECT fact FROM customer_facts WHERE customer_id = %s"
                " ORDER BY created_at DESC LIMIT %s",
                (customer_id, limit),
            ).fetchall()
        return [row[0] for row in rows]


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


# ==================================================================== 门面


class ProfileService:
    def __init__(self, store: Any = None) -> None:
        if store is not None:
            self.store = store
        else:
            factory = get_postgres()
            self.store = PostgresProfileStore(factory) if factory else SQLiteProfileStore()

    @property
    def backend(self) -> str:
        return getattr(self.store, "name", "?")

    # ------------------------------------------------------------- 偏好
    def set(self, customer_id: str, key: str, value: Any, source: str = "agent") -> None:
        self.store.set_preference(customer_id, key, value, source)

    def get(self, customer_id: str, key: str, default: Any = None) -> Any:
        return self.store.all_preferences(customer_id).get(key, default)

    def all_preferences(self, customer_id: str) -> dict[str, Any]:
        return self.store.all_preferences(customer_id)

    def add_fact(self, customer_id: str, fact: str, run_id: str = "") -> None:
        self.store.add_fact(customer_id, fact, run_id)

    def recent_facts(self, customer_id: str, limit: int = 5) -> list[str]:
        return self.store.recent_facts(customer_id, limit)

    # ------------------------------------------------------------- 学习
    def learn_from_run(
        self,
        customer_id: str,
        intent: str,
        outcome: dict[str, Any],
        run_id: str = "",
    ) -> list[str]:
        """把一次处理的结论沉淀成长期偏好。

        这是"长期偏好"能真正生效的关键一步：没有它，偏好表永远是空的，
        所谓"PostgreSQL 存长期用户偏好"就只是简历上的一句话。
        """
        learned: list[str] = []
        if not customer_id:
            return learned

        resolution = outcome.get("resolution", "")
        if resolution:
            self.set(customer_id, "上次结案方式", resolution, source=f"run:{run_id}")
            learned.append(f"上次结案方式={resolution}")

        channel = outcome.get("channel")
        if channel:
            self.set(customer_id, "偏好通知渠道", channel, source=f"run:{run_id}")
            learned.append(f"偏好通知渠道={channel}")

        for tag in outcome.get("tags", []) or []:
            self.set(customer_id, f"标记:{tag}", True, source=f"run:{run_id}")
            learned.append(f"标记:{tag}")

        if outcome.get("needs_manual"):
            self.set(customer_id, "需人工复核", True, source=f"run:{run_id}")
            learned.append("需人工复核=True")

        if outcome.get("summary"):
            self.add_fact(customer_id, f"[{intent}] {outcome['summary']}", run_id)

        return learned


_service: ProfileService | None = None


def profile_service() -> ProfileService:
    global _service
    if _service is None:
        _service = ProfileService()
    return _service


def reset_service() -> None:
    global _service
    _service = None
