"""Checkpointer：把图的运行状态存下来，让它能"接着跑"。

这是 HITL（人机协同）能成立的前提：
  审批可能发生在 3 秒后（客服顺手点一下），也可能在 3 小时后（夜班交接），
  中间 Web 服务可能重启。所以状态必须**离开进程**。

三种实现，按可用性自动挑选：
  · RedisCheckpointer   —— 生产首选：快、带 TTL、天然支持多实例共享
  · FileCheckpointer    —— 单机持久化（JSON 落盘），本项目默认，重启不丢
  · MemoryCheckpointer  —— 测试用，进程结束即消失
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import settings


class MemoryCheckpointer:
    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    def save(self, thread_id: str, payload: dict) -> None:
        self._data[thread_id] = payload

    def load(self, thread_id: str) -> dict | None:
        return self._data.get(thread_id)

    def delete(self, thread_id: str) -> None:
        self._data.pop(thread_id, None)

    def list_threads(self) -> list[str]:
        return list(self._data)


class FileCheckpointer:
    """每个会话一个 JSON 文件：人肉可读，排障时可以直接打开看卡在哪。"""

    name = "file"

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or settings.state_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in thread_id)
        return self.dir / f"{safe}.json"

    def save(self, thread_id: str, payload: dict) -> None:
        path = self._path(thread_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)          # 原子替换：避免写到一半进程挂掉留下半个文件

    def load(self, thread_id: str) -> dict | None:
        path = self._path(thread_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def delete(self, thread_id: str) -> None:
        self._path(thread_id).unlink(missing_ok=True)

    def list_threads(self) -> list[str]:
        return [p.stem for p in self.dir.glob("*.json")]

    def purge_older_than(self, seconds: int) -> int:
        """清理过期存档，防止磁盘无限增长。"""
        cutoff = time.time() - seconds
        removed = 0
        for path in self.dir.glob("*.json"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        return removed


class RedisCheckpointer:
    """生产实现：多实例共享 + TTL 自动过期。"""

    name = "redis"

    def __init__(self, client: Any, prefix: str = "aftersale:ckpt:", ttl: int = 86400) -> None:
        self.client = client
        self.prefix = prefix
        self.ttl = ttl

    def save(self, thread_id: str, payload: dict) -> None:
        self.client.setex(
            f"{self.prefix}{thread_id}", self.ttl, json.dumps(payload, ensure_ascii=False)
        )

    def load(self, thread_id: str) -> dict | None:
        raw = self.client.get(f"{self.prefix}{thread_id}")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def delete(self, thread_id: str) -> None:
        self.client.delete(f"{self.prefix}{thread_id}")

    def list_threads(self) -> list[str]:
        return [k.decode().replace(self.prefix, "") if isinstance(k, bytes)
                else k.replace(self.prefix, "")
                for k in self.client.keys(f"{self.prefix}*")]


class ChainCheckpointer:
    """组合：写 Redis（快、共享），同时落盘（Redis 挂了也不丢）。

    "先写缓存再写盘"看起来多此一举，但对**退款审批**这种场景，
    状态丢一次就意味着客户多等一小时 —— 冗余是值得的。
    """

    name = "chain"

    def __init__(self, primary: Any, secondary: Any) -> None:
        self.primary = primary
        self.secondary = secondary

    def save(self, thread_id: str, payload: dict) -> None:
        for target in (self.primary, self.secondary):
            try:
                target.save(thread_id, payload)
            except Exception:  # noqa: BLE001 —— 任一层失败不影响主流程
                continue

    def load(self, thread_id: str) -> dict | None:
        for target in (self.primary, self.secondary):
            try:
                found = target.load(thread_id)
            except Exception:  # noqa: BLE001
                found = None
            if found:
                return found
        return None

    def delete(self, thread_id: str) -> None:
        for target in (self.primary, self.secondary):
            try:
                target.delete(thread_id)
            except Exception:  # noqa: BLE001
                continue

    def list_threads(self) -> list[str]:
        try:
            return self.primary.list_threads()
        except Exception:  # noqa: BLE001
            return self.secondary.list_threads()


def build_checkpointer() -> Any:
    """按可用性挑：Redis 可用 → Redis+文件；否则 → 文件。"""
    from ..runtime import get_redis

    file_store = FileCheckpointer()
    client = get_redis()
    if client is not None:
        return ChainCheckpointer(RedisCheckpointer(client), file_store)
    return file_store
