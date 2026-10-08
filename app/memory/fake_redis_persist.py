"""fakeredis 的跨进程持久化垫片。

问题：
  未部署真 Redis 时，`fakeredis.FakeRedis()` 的数据只存在于**当前进程的内存**。
  于是跨进程场景会失真：
    · 进程 A 跑出"待审批"状态 → 存进 fakeredis
    · 进程 B 去批准 → fakeredis 是空的 → 报"没有找到执行存档"
  这会让"跨进程中断恢复"这个卖点在演示时**看起来是坏的**（其实是环境问题）。

做法：
  在 fakeredis 外面套一层"写穿 + 启动回载"：
    · 每次写命令后，把受管前缀下的 key 快照到本地文件（JSON）
    · 进程启动时（首次取连接），把快照加载回 fakeredis
  文件格式刻意做得可读，排障时能直接打开看。

边界（必须诚实）：
  这不是"分布式 Redis"。它只解决**单机跨进程**的持久化，
  不支持多实例并发写、不支持原子跨进程事务。
  真上生产仍然是 `REDIS_URL` 指向真 Redis —— 那时这层垫片完全不参与。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# 需要跨进程持久化的 key 前缀（业务上都是这些）
_MANAGED_PREFIXES = ("aftersale:session:", "aftersale:ckpt:")


class PersistedFakeRedis:
    """FakeRedis 代理：写穿到 JSON 文件，启动时回载。

    只拦截需要落盘的写命令（set/setex/setnx/delete/expire），
    其余命令原样转发 —— 不做全量实现，避免行为漂移。
    """

    _WRITE_CMDS = {"set", "setex", "psetex", "setnx", "delete", "unlink", "expire", "persist"}

    def __init__(self, inner: Any, path: Path) -> None:
        self._inner = inner
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._last_snapshot = 0.0
        self._restore()

    # ------------------------------------------------------------ 快照
    def _managed_keys(self) -> list[str]:
        found: list[str] = []
        for prefix in _MANAGED_PREFIXES:
            for key in self._inner.keys(f"{prefix}*"):
                found.append(key if isinstance(key, str) else key.decode())
        return found

    def _restore(self) -> None:
        """启动时把磁盘快照灌回 fakeredis。"""
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for key, item in (data.get("kv") or {}).items():
            value = item.get("value")
            ttl = item.get("ttl")
            try:
                if ttl and ttl > 0:
                    self._inner.setex(key, int(ttl), value)
                else:
                    self._inner.set(key, value)
            except Exception:  # noqa: BLE001
                continue

    def _snapshot(self) -> None:
        """把受管 key 写到磁盘。节流：1 秒内最多写一次，避免高频 IO。"""
        now = time.time()
        if now - self._last_snapshot < 1.0:
            return
        self._last_snapshot = now
        kv: dict[str, Any] = {}
        for key in self._managed_keys():
            try:
                value = self._inner.get(key)
                ttl = self._inner.ttl(key)
            except Exception:  # noqa: BLE001
                continue
            if value is None:
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            kv[key] = {"value": value, "ttl": int(ttl) if ttl and ttl > 0 else None}
        payload = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "kv": kv}
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass

    # ------------------------------------------------------------ 命令转发
    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name in self._WRITE_CMDS:
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # 强刷：写命令后立刻快照，保证"写进 Redis"= "进程死了也不丢"
                self._last_snapshot = 0.0
                self._snapshot()
                return result

            return wrapper
        return attr

    # 让 isinstance/类型判断与排障更直观
    def __repr__(self) -> str:
        return f"PersistedFakeRedis(path={self._path.name})"
