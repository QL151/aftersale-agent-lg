"""运行时能力探测：决定每个可选依赖"用真的还是用降级实现"。

这是本项目能同时满足两个矛盾要求的关键：
  ① 简历上写的技术栈（LangGraph / Redis / PostgreSQL / Langfuse / MCP / Docker）**必须真接**
  ② 但任何人都应该能 `python run_cli.py` 直接跑起来（不需要装中间件）

做法：接口 + 适配器 + 探测。
  每个外部依赖都定义成一个窄接口，有两个实现：
    · 真实现（import 得到就用）
    · 降级实现（内存 / SQLite / 本地 JSONL，语义一致）
  探测结果会被缓存，并通过 `doctor()` 一次性打印出来 ——
  面试演示时第一件事就是跑它，让人一眼看到"哪些是真接了、哪些环境缺"。

**这条"降级"设计本身就是加分项**：它说明你考虑过部署环境差异，
而不是写了个只能在你笔记本上跑的 demo。
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

from .config import settings

_cache: dict[str, Any] = {}


def _has_module(name: str) -> bool:
    if name in _cache:
        return bool(_cache[name])
    found = importlib.util.find_spec(name) is not None
    _cache[name] = found
    return found


def reset_cache() -> None:
    _cache.clear()


# ==================================================================== Redis


def get_redis() -> Any:
    """返回可用的 redis 客户端；不可用返回 None（调用方走内存降级）。

    三级探测：
      ① 真 Redis（redis://... 连得上）      → 生产语义，多实例共享 + TTL
      ② fakeredis（未部署 Redis 时的替身）  → 同一套命令语义，进程内存储
      ③ 都没有                              → None，调用方降级为内存 + JSONL

    ②的存在是为了让"Redis 这条链路"在**任何机器上都能真跑一遍**：
    演示、CI、面试现场都不用先装 Redis 服务。它和真 Redis 的区别只在
    "存储在哪"（进程内存 vs 独立进程），命令行为一致。
    """
    if "redis_client" in _cache:
        return _cache["redis_client"]
    client = None
    if settings.redis_enabled and _has_module("redis"):
        try:
            import redis  # type: ignore[import-not-found]

            candidate = redis.Redis.from_url(
                settings.redis_url, decode_responses=True,
                socket_connect_timeout=1.5, socket_timeout=1.5,
            )
            candidate.ping()
            client = candidate
        except Exception:  # noqa: BLE001 —— 连不上就降级，不要让整个服务起不来
            client = None

        # 真 Redis 连不上 → 用 fakeredis 顶上（保持"Redis 真跑"而不是完全降级）
        if client is None and settings.redis_fake_fallback and _has_module("fakeredis"):
            try:
                import fakeredis  # type: ignore[import-not-found]

                from .memory.fake_redis_persist import PersistedFakeRedis

                fake = fakeredis.FakeRedis(decode_responses=True)
                fake.ping()
                # 套一层写穿持久化：fakeredis 本身是纯内存的，
                # 不落盘就没法跨进程（"待审批 → 重启 → 批准"这条链路会断）。
                client = PersistedFakeRedis(fake, settings.state_dir / "fake_redis.json")
                _cache["redis_is_fake"] = True
            except Exception:  # noqa: BLE001
                client = None
    _cache["redis_client"] = client
    return client


def redis_is_fake() -> bool:
    """当前用的是不是 fakeredis 替身（doctor 里要如实标注）。"""
    get_redis()
    return bool(_cache.get("redis_is_fake"))


def redis_status() -> str:
    if not settings.redis_enabled:
        return "disabled(配置关闭)"
    if not _has_module("redis"):
        return "fallback(未安装 redis 包 → 用内存会话记忆)"
    client = get_redis()
    if client is None:
        return f"fallback(连不上 {settings.redis_url} → 用内存会话记忆)"
    if redis_is_fake():
        return "ok(fakeredis 内存替身 → 命令语义与真 Redis 一致)"
    return "ok(真实 Redis 服务)"


# ==================================================================== PostgreSQL


def get_postgres() -> Any:
    """返回一个"能开连接"的工厂；不可用返回 None（调用方走 SQLite 降级）。"""
    if "pg_factory" in _cache:
        return _cache["pg_factory"]
    factory = None
    if settings.postgres_enabled and _has_module("psycopg"):
        try:
            import psycopg  # type: ignore[import-not-found]

            conn = psycopg.connect(settings.postgres_dsn, connect_timeout=2)
            conn.close()
            factory = lambda: psycopg.connect(settings.postgres_dsn)  # noqa: E731
        except Exception:  # noqa: BLE001
            factory = None
    _cache["pg_factory"] = factory
    return factory


def postgres_status() -> str:
    if not settings.postgres_enabled:
        return "disabled(配置关闭)"
    if not _has_module("psycopg"):
        return "fallback(未安装 psycopg → 用 SQLite 存长期偏好)"
    return "ok" if get_postgres() is not None else "fallback(连不上 → 用 SQLite 存长期偏好)"


# ==================================================================== Langfuse


def get_langfuse() -> Any:
    if "langfuse" in _cache:
        return _cache["langfuse"]
    client = None
    if settings.langfuse_enabled and _has_module("langfuse") and settings.langfuse_public_key:
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]

            client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:  # noqa: BLE001
            client = None
    _cache["langfuse"] = client
    return client


def langfuse_status() -> str:
    if not settings.langfuse_enabled:
        return "disabled(配置关闭)"
    if not _has_module("langfuse"):
        return "fallback(未安装 langfuse → Trace 写本地 JSONL，数据形状与 Langfuse 对齐)"
    if not settings.langfuse_public_key:
        return "fallback(未配置 Key → Trace 写本地 JSONL)"
    return "ok" if get_langfuse() is not None else "fallback(初始化失败 → 本地 JSONL)"


# ==================================================================== LangGraph


def langgraph_available() -> bool:
    return _has_module("langgraph")


def engine_status() -> str:
    requested = settings.graph_engine
    if requested == "native":
        return "native(配置强制自研引擎)"
    if requested == "langgraph":
        return "langgraph(配置强制)" if langgraph_available() else "native(配置要 langgraph 但未安装 → 回退自研)"
    return "langgraph(自动探测)" if langgraph_available() else "native(未安装 langgraph → 自研等价引擎)"


# ==================================================================== MCP


def mcp_status() -> str:
    if not settings.mcp_servers:
        return "disabled(未配置 MCP_SERVERS)"
    return f"configured({len(settings.mcp_servers.split(','))} 个 server)"


# ==================================================================== 沙箱


def sandbox_status() -> str:
    if not settings.sandbox_enabled:
        return "disabled(配置关闭)"
    from .sandbox import _docker_available  # 复用探测逻辑

    return "docker(真隔离)" if _docker_available() else "subprocess(进程+AST 校验，非安全边界)"


# ==================================================================== 体检报告


@dataclass
class Capability:
    name: str
    status: str
    detail: str

    @property
    def real(self) -> bool:
        return self.status.startswith(("ok", "docker", "langgraph", "configured"))


def doctor() -> list[Capability]:
    """一次性报告所有依赖的真实状态。演示和排障的第一条命令。"""
    llm = (
        f"mock(未配置 Key，用离线假模型：{settings.llm_model})"
        if settings.use_mock
        else f"ok({settings.llm_model} @ {settings.llm_base_url})"
    )
    return [
        Capability("大模型", llm, "Function Calling 的工具调用方"),
        Capability("状态机引擎", engine_status(), "LangGraph 或自研等价引擎"),
        Capability("会话记忆 (Redis)", redis_status(), "短期多轮上下文 + 槽位记忆"),
        Capability("长期偏好 (PostgreSQL)", postgres_status(), "客户偏好与画像沉淀"),
        Capability("链路追踪 (Langfuse)", langfuse_status(), "Trace / Bad Case 定位"),
        Capability("MCP", mcp_status(), "把内部系统封装成标准工具"),
        Capability("代码沙箱", sandbox_status(), "数据分析类代码执行"),
        Capability("业务库", "ok(SQLite)",
                  f"{settings.db_path.name}：订单/物流/工单/审批单"),
    ]


def print_doctor() -> None:
    print("=" * 88)
    print("运行时能力体检（真接的显示 ok，环境缺失的显示 fallback —— 不影响跑通）")
    print("=" * 88)
    for item in doctor():
        mark = "[真实]" if item.real else "[降级]"
        print(f"  {mark} {item.name:<24} {item.status}")
        print(f"         └─ {item.detail}")
    print("=" * 88)
