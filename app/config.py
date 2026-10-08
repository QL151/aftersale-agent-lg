"""集中配置：一切可变项都来自环境变量 / .env，代码里不写死。

设计要点（面试会问）：
  本项目所有外部依赖（Redis / PostgreSQL / Langfuse / LangGraph / MCP / Docker 沙箱）
  都是**可选**的。配置层只回答"用户想不想用"，真正的可用性由各适配器在运行时探测
  （见 app/runtime.py 的 doctor()）。
  这样同一条代码路径既能跑在"零依赖笔记本"上，也能跑在"生产全套中间件"上 ——
  这是项目能离线演示、又能写进简历的关键。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # 没装 python-dotenv 也能跑
    def load_dotenv(*_args, **_kwargs):
        return False


BASE_DIR = Path(__file__).resolve().parent.parent


def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    return _str(name, "1" if default else "0").lower() in {"1", "true", "yes", "y", "on"}


def _path(name: str, default: str) -> Path:
    value = Path(_str(name, default))
    return value if value.is_absolute() else (BASE_DIR / value)


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------ 大模型
    llm_api_key: str
    llm_base_url: str
    llm_model: str                 # 主模型（用来做规划和长尾推理）
    llm_small_model: str           # 降级小模型（成本熔断后切到它）
    llm_temperature: float
    llm_timeout: float
    mock_llm: bool                 # True = 离线假模型，不需要 Key、不花钱、可复现

    # ------------------------------------------------------------ Agent 行为 / 状态机
    graph_engine: str              # auto / langgraph / native
    max_steps: int                 # 单次运行最多推进几个节点（防无限循环）
    max_tool_calls: int            # 单次运行最多调用几次工具
    max_total_seconds: float       # 单次运行总时长上限
    max_same_call: int             # 同工具同参数最多重复几次
    max_node_retries: int          # 节点失败重试次数（失败回退）
    confirm_mode: str              # ask / allow / deny —— 危险操作确认策略

    # ------------------------------------------------------------ 意图路由
    router_confidence_threshold: float   # 低于它 → 判定长尾，转自主 Agent
    router_use_llm: bool                 # 有 Key 时是否用模型兜底分类

    # ------------------------------------------------------------ 成本熔断
    session_token_budget: int      # 单会话 token 上限
    run_token_budget: int          # 单次运行 token 上限
    cost_warn_ratio: float         # 用到这个比例开始告警
    price_prompt_per_1k: float     # 主模型单价（元 / 1k prompt token）
    price_completion_per_1k: float
    price_small_per_1k: float      # 降级模型单价
    cost_downgrade: bool           # 超预算是否自动降级到小模型

    # ------------------------------------------------------------ 会话记忆 / 长期偏好
    redis_url: str
    redis_enabled: bool
    redis_fake_fallback: bool      # 真 Redis 连不上时，用 fakeredis 内存替身顶上
    session_ttl_seconds: int
    postgres_dsn: str
    postgres_enabled: bool

    # ------------------------------------------------------------ 观测
    langfuse_enabled: bool
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str

    # ------------------------------------------------------------ 人工协同
    approval_timeout_seconds: int  # 审批超时
    approval_timeout_action: str   # escalate（转人工工单）/ reject（拒绝）
    approval_auto_approve_max: float  # 低于这个金额自动放行（小额免打扰）

    # ------------------------------------------------------------ 沙箱 / MCP
    sandbox_enabled: bool
    sandbox_timeout: float
    sandbox_max_output: int
    mcp_servers: str               # 逗号分隔的 MCP server 启动命令

    # ------------------------------------------------------------ 存储
    db_path: Path                  # 业务库（订单/物流/工单）
    trace_dir: Path
    state_dir: Path                # checkpointer / 审批队列 / 会话落盘

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def use_mock(self) -> bool:
        return self.mock_llm or not self.llm_ready

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BASE_DIR / ".env")
        return cls(
            llm_api_key=_str("LLM_API_KEY"),
            llm_base_url=_str("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            llm_model=_str("LLM_MODEL", "deepseek-chat"),
            llm_small_model=_str("LLM_SMALL_MODEL", "qwen-turbo"),
            llm_temperature=_float("LLM_TEMPERATURE", 0.0),
            llm_timeout=_float("LLM_TIMEOUT", 60.0),
            mock_llm=_bool("MOCK_LLM", False),
            graph_engine=_str("GRAPH_ENGINE", "auto").lower(),
            max_steps=_int("AGENT_MAX_STEPS", 12),
            max_tool_calls=_int("AGENT_MAX_TOOL_CALLS", 12),
            max_total_seconds=_float("AGENT_MAX_SECONDS", 120.0),
            max_same_call=_int("AGENT_MAX_SAME_CALL", 2),
            max_node_retries=_int("AGENT_MAX_NODE_RETRIES", 1),
            confirm_mode=_str("AGENT_CONFIRM_MODE", "ask").lower(),
            router_confidence_threshold=_float("ROUTER_CONFIDENCE_THRESHOLD", 0.62),
            router_use_llm=_bool("ROUTER_USE_LLM", True),
            session_token_budget=_int("SESSION_TOKEN_BUDGET", 60000),
            run_token_budget=_int("RUN_TOKEN_BUDGET", 20000),
            cost_warn_ratio=_float("COST_WARN_RATIO", 0.8),
            price_prompt_per_1k=_float("PRICE_PROMPT_PER_1K", 0.001),
            price_completion_per_1k=_float("PRICE_COMPLETION_PER_1K", 0.002),
            price_small_per_1k=_float("PRICE_SMALL_PER_1K", 0.0002),
            cost_downgrade=_bool("COST_DOWNGRADE", True),
            redis_url=_str("REDIS_URL", "redis://127.0.0.1:6379/0"),
            redis_enabled=_bool("REDIS_ENABLED", True),
            redis_fake_fallback=_bool("REDIS_FAKE_FALLBACK", True),
            session_ttl_seconds=_int("SESSION_TTL_SECONDS", 3600),
            postgres_dsn=_str("POSTGRES_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/aftersale"),
            postgres_enabled=_bool("POSTGRES_ENABLED", True),
            langfuse_enabled=_bool("LANGFUSE_ENABLED", True),
            langfuse_public_key=_str("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=_str("LANGFUSE_SECRET_KEY"),
            langfuse_host=_str("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            approval_timeout_seconds=_int("APPROVAL_TIMEOUT_SECONDS", 1800),
            approval_timeout_action=_str("APPROVAL_TIMEOUT_ACTION", "escalate").lower(),
            # 默认 200 元以下、且客户风险不高的退款自动放行（"小额免打扰"）。
            # 这不是放松安全，而是把审核人力留给真正有风险的工单 —— 否则客服会被确认框淹没。
            approval_auto_approve_max=_float("APPROVAL_AUTO_APPROVE_MAX", 200.0),
            sandbox_enabled=_bool("SANDBOX_ENABLED", True),
            sandbox_timeout=_float("SANDBOX_TIMEOUT", 5.0),
            sandbox_max_output=_int("SANDBOX_MAX_OUTPUT", 4000),
            mcp_servers=_str("MCP_SERVERS"),
            db_path=_path("DB_PATH", "data/aftersale.db"),
            trace_dir=_path("TRACE_DIR", "data/traces"),
            state_dir=_path("STATE_DIR", "data/state"),
        )

    def ensure_dirs(self) -> None:
        for path in (self.db_path.parent, self.trace_dir, self.state_dir):
            path.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        """给 CLI / 接口用的安全信息（不含任何密钥）。"""
        return {
            "mock_llm": self.use_mock,
            "llm_model": self.llm_model if self.llm_ready else "（未配置 Key，将使用离线假模型）",
            "llm_small_model": self.llm_small_model,
            "graph_engine": self.graph_engine,
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "confirm_mode": self.confirm_mode,
            "router_threshold": self.router_confidence_threshold,
            "session_token_budget": self.session_token_budget,
        }


settings = Settings.from_env()
