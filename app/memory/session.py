"""会话记忆（短期）：多轮对话上下文 + 槽位记忆。

为什么需要它？
  售后对话天然是多轮的：
    用户：「我要退款」→ Agent：「请提供订单号」→ 用户：「SO2026000123」
  如果每一轮都从零开始，Agent 就会反复问同一个问题 —— 这正是"平均交互轮次"
  居高不下的主要原因。会话记忆把已确认的槽位（订单号、客户、意图）带着走。

两种实现（自动挑选）：
  · RedisSessionStore  —— 生产：多实例共享、带 TTL 自动过期
  · LocalSessionStore  —— 降级：内存 + JSONL 落盘，重启不丢，零依赖可跑

一个刻意的取舍：**上下文不是越长越好**。
  历史超过阈值就做摘要压缩（compress），只保留"事实 + 未完成事项"，
  因为把 20 轮闲聊塞进 prompt 会既贵又让模型跑偏。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import settings
from ..runtime import get_redis


@dataclass
class SessionMemory:
    session_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)   # [{role, content, ts}]
    slots: dict[str, Any] = field(default_factory=dict)         # 已确认的业务槽位
    summary: str = ""                                           # 压缩后的历史摘要
    usage: dict[str, float] = field(default_factory=dict)       # 会话累计用量（成本熔断用）
    updated_at: float = 0.0

    def to_payload(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": self.turns,
            "slots": self.slots,
            "summary": self.summary,
            "usage": self.usage,
            "updated_at": self.updated_at or time.time(),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "SessionMemory":
        return cls(
            session_id=payload.get("session_id", ""),
            turns=payload.get("turns", []) or [],
            slots=payload.get("slots", {}) or {},
            summary=payload.get("summary", "") or "",
            usage=payload.get("usage", {}) or {},
            updated_at=payload.get("updated_at", 0.0) or 0.0,
        )

    # ---------------------------------------------------------- 槽位
    def merge_slots(self, slots: dict[str, Any]) -> None:
        """只接受非空值，避免"这轮没说"把上一轮确认的订单号冲掉。"""
        for key, value in (slots or {}).items():
            if value not in (None, "", [], {}):
                self.slots[key] = value

    def slot(self, key: str, default: Any = None) -> Any:
        return self.slots.get(key, default)

    # ---------------------------------------------------------- 上下文
    def context_messages(self, limit: int = 8) -> list[dict[str, str]]:
        """拼给模型的对话上下文：摘要（如果有）+ 最近若干轮。"""
        messages: list[dict[str, str]] = []
        if self.summary:
            messages.append({"role": "system", "content": f"【历史摘要】{self.summary}"})
        for turn in self.turns[-limit:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        return messages

    def compress(self, keep: int = 4, max_chars: int = 400) -> None:
        """历史过长时把前半段压成一句摘要（不调模型，规则压缩，零成本）。"""
        if len(self.turns) <= keep * 2:
            return
        old, recent = self.turns[:-keep], self.turns[-keep:]
        facts: list[str] = []
        for turn in old:
            text = (turn.get("content") or "").strip().replace("\n", " ")
            if text:
                facts.append(text[:60])
        merged = (self.summary + " " if self.summary else "") + "；".join(facts[-6:])
        self.summary = merged[-max_chars:]
        self.turns = recent


# ==================================================================== 存储实现


class LocalSessionStore:
    """内存 + JSONL 落盘。零依赖、重启不丢，适合单机演示与自动化测试。"""

    name = "local"

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or settings.state_dir / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, SessionMemory] = {}

    def _path(self, session_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)
        return self.dir / f"{safe}.json"

    def load(self, session_id: str) -> SessionMemory:
        if session_id in self._mem:
            return self._mem[session_id]
        path = self._path(session_id)
        if path.exists():
            try:
                memory = SessionMemory.from_payload(json.loads(path.read_text(encoding="utf-8")))
                self._mem[session_id] = memory
                return memory
            except (json.JSONDecodeError, OSError):
                pass
        memory = SessionMemory(session_id=session_id)
        self._mem[session_id] = memory
        return memory

    def save(self, memory: SessionMemory) -> None:
        memory.updated_at = time.time()
        self._mem[memory.session_id] = memory
        self._path(memory.session_id).write_text(
            json.dumps(memory.to_payload(), ensure_ascii=False), encoding="utf-8"
        )

    def clear(self, session_id: str) -> None:
        self._mem.pop(session_id, None)
        self._path(session_id).unlink(missing_ok=True)


class RedisSessionStore:
    """生产实现：跨实例共享 + TTL 自动过期。"""

    name = "redis"

    def __init__(self, client: Any, prefix: str = "aftersale:session:", ttl: int | None = None) -> None:
        self.client = client
        self.prefix = prefix
        self.ttl = ttl or settings.session_ttl_seconds

    def load(self, session_id: str) -> SessionMemory:
        raw = self.client.get(f"{self.prefix}{session_id}")
        if raw is None:
            return SessionMemory(session_id=session_id)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return SessionMemory.from_payload(json.loads(raw))
        except json.JSONDecodeError:
            return SessionMemory(session_id=session_id)

    def save(self, memory: SessionMemory) -> None:
        memory.updated_at = time.time()
        self.client.setex(
            f"{self.prefix}{memory.session_id}", self.ttl,
            json.dumps(memory.to_payload(), ensure_ascii=False),
        )

    def clear(self, session_id: str) -> None:
        self.client.delete(f"{self.prefix}{session_id}")


# ==================================================================== 门面


class SessionMemoryService:
    """业务层只跟它打交道，感知不到底层是 Redis 还是本地文件。"""

    def __init__(self, store: Any = None) -> None:
        if store is not None:
            self.store = store
        else:
            client = get_redis()
            self.store = RedisSessionStore(client) if client is not None else LocalSessionStore()

    @property
    def backend(self) -> str:
        return getattr(self.store, "name", "?")

    # ------------------------------------------------------------- 读写
    def load(self, session_id: str) -> SessionMemory:
        return self.store.load(session_id)

    def append_turn(self, session_id: str, role: str, content: str) -> SessionMemory:
        memory = self.store.load(session_id)
        memory.turns.append({"role": role, "content": content, "ts": time.time()})
        memory.compress()
        self.store.save(memory)
        return memory

    def remember_slots(self, session_id: str, **slots: Any) -> SessionMemory:
        memory = self.store.load(session_id)
        memory.merge_slots(slots)
        self.store.save(memory)
        return memory

    def forget(self, session_id: str, *keys: str) -> None:
        """删除槽位（merge_slots 会忽略空值，所以清理必须走这里）。"""
        memory = self.store.load(session_id)
        for key in keys:
            memory.slots.pop(key, None)
        self.store.save(memory)

    def clear(self, session_id: str) -> None:
        self.store.clear(session_id)

    def snapshot(self, session_id: str) -> dict:
        memory = self.store.load(session_id)
        return {
            "session_id": session_id,
            "backend": self.backend,
            "turns": len(memory.turns),
            "slots": memory.slots,
            "summary": memory.summary,
        }


_service: SessionMemoryService | None = None


def session_service() -> SessionMemoryService:
    """全局单例：避免每次调用都重新探测 Redis。"""
    global _service
    if _service is None:
        _service = SessionMemoryService()
    return _service


def reset_service() -> None:
    """仅测试用。"""
    global _service
    _service = None
