"""工具注册中心：声明式定义工具，自动生成给模型看的描述，并做本地参数校验。

相比一个"玩具 Agent"，这里多了四件在生产里必须有的事：

  1. **风险分级**（risk_level = low/medium/high）
     不是所有"危险"都一个待遇：查订单是 low，发券是 medium，动钱是 high。
     人机协同（HITL）按等级决定策略，而不是一刀切弹窗 —— 否则客服会被确认框淹没。

  2. **上下文注入**（_ctx: ToolContext）
     工具需要知道"这次调用属于哪个 run / 哪个会话"，才能做幂等和审计。
     但**绝不能让模型自己生成幂等键** —— 模型每次生成的不一样，幂等就失效了。
     所以由运行时注入，模型看不到这个参数。

  3. **工具清单裁剪**（specs_for(tags)）
     把 19 个工具全塞进 prompt 既贵又容易让模型选错。
     确定性工作流只暴露它需要的那 2~4 个；长尾才放开全量。

  4. **结果截断**
     工具返回一大坨文本会直接吃掉上下文预算，所以统一截断 + 标注。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


# ==================================================================== 上下文


@dataclass
class ToolContext:
    """运行时注入给工具的上下文（模型看不见、也不能伪造）。"""

    run_id: str = ""
    thread_id: str = ""
    session_id: str = ""
    intent: str = ""
    repo: Any = None                 # 业务库仓储
    profile: Any = None              # 长期偏好存储
    session_memory: Any = None       # 会话记忆
    extra: dict[str, Any] = field(default_factory=dict)

    def idempotency_key(self, action: str, *parts: Any) -> str:
        """由**系统**推导幂等键：同一次运行里同样的动作 = 同一个键。

        这样"模型重试 / 图节点重跑 / 网络超时后重发"都不会重复退款，
        而不需要模型理解什么叫幂等。
        """
        joined = "|".join(str(p) for p in parts)
        return f"{action}:{self.run_id}:{joined}"


# ==================================================================== 工具


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    dangerous: bool = False            # 有副作用（会改动数据 / 动钱）
    risk_level: str = "low"            # low / medium / high
    timeout: float = 10.0
    policy: str = ""                   # 内容级策略名，由 Guardrail 解释
    tags: tuple[str, ...] = ()         # 场景标签：order / logistics / refund / ticket ...
    idempotent: bool = False           # 重复执行是否无害
    inject_context: bool = False       # 是否需要运行时注入 _ctx
    max_output: int = 2000             # 返回给模型的最大字符数

    def to_llm_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # ------------------------------------------------------------ 参数校验
    def validate(self, arguments: dict) -> str | None:
        """JSON Schema 子集校验。返回错误说明；None = 通过。"""
        schema = self.parameters or {}
        properties: dict = schema.get("properties", {})
        required: list = schema.get("required", [])

        for key in required:
            if key not in arguments or arguments[key] in (None, ""):
                return f"缺少必填参数 `{key}`"

        for key, value in arguments.items():
            spec = properties.get(key)
            if spec is None:
                if schema.get("additionalProperties") is False:
                    return f"不认识的参数 `{key}`"
                continue

            expected = spec.get("type")
            if expected and not _type_ok(value, expected):
                return f"参数 `{key}` 类型应为 {expected}，实际收到 {type(value).__name__}"

            if "enum" in spec and value not in spec["enum"]:
                return f"参数 `{key}` 只能是 {spec['enum']} 之一，收到 {value!r}"

            if expected in {"number", "integer"}:
                if "minimum" in spec and value < spec["minimum"]:
                    return f"参数 `{key}` 不能小于 {spec['minimum']}（收到 {value}）"
                if "maximum" in spec and value > spec["maximum"]:
                    return f"参数 `{key}` 不能大于 {spec['maximum']}（收到 {value}）"

            if expected == "string":
                if "minLength" in spec and len(value) < spec["minLength"]:
                    return f"参数 `{key}` 太短（至少 {spec['minLength']} 个字符）"
                if "maxLength" in spec and len(value) > spec["maxLength"]:
                    return f"参数 `{key}` 太长（最多 {spec['maxLength']} 个字符）"

            if expected == "array" and "maxItems" in spec and len(value) > spec["maxItems"]:
                return f"参数 `{key}` 最多 {spec['maxItems']} 项"
        return None

    def call(self, arguments: dict, ctx: ToolContext | None = None) -> str:
        """执行工具：按需注入上下文，并截断过长输出。"""
        kwargs = dict(arguments)
        if self.inject_context:
            kwargs["_ctx"] = ctx or ToolContext()
        raw = self.func(**kwargs)
        text = raw if isinstance(raw, str) else str(raw)
        if len(text) > self.max_output:
            text = text[: self.max_output] + f"\n…（输出已截断，共 {len(text)} 字符）"
        return text


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    python_type = _TYPE_MAP.get(expected)
    return python_type is None or isinstance(value, python_type)


# ==================================================================== 注册表

_REGISTRY: dict[str, Tool] = {}
_LOADED = False
_MCP_ATTACHED = False


def tool(
    name: str,
    description: str,
    parameters: dict | None = None,
    dangerous: bool = False,
    risk_level: str = "low",
    timeout: float = 10.0,
    policy: str = "",
    tags: tuple[str, ...] = (),
    idempotent: bool = False,
    max_output: int = 2000,
):
    """装饰器：把一个普通函数注册成 Agent 可调用的工具。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"工具名重复：{name}")
        wants_ctx = "_ctx" in inspect.signature(func).parameters
        _REGISTRY[name] = Tool(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            func=func,
            dangerous=dangerous,
            risk_level=risk_level,
            timeout=timeout,
            policy=policy,
            tags=tags,
            idempotent=idempotent,
            inject_context=wants_ctx,
            max_output=max_output,
        )
        return func

    return decorator


def all_tools() -> list[Tool]:
    return list(_REGISTRY.values())


def register(tool_obj: Tool, override: bool = False) -> bool:
    """运行时动态注册工具（MCP 工具发现就靠它）。

    返回是否注册成功；重名且 override=False 时返回 False —— 本地工具优先，
    不让远端 MCP Server 覆盖掉经过验证的本地实现。
    """
    if tool_obj.name in _REGISTRY and not override:
        return False
    _REGISTRY[tool_obj.name] = tool_obj
    return True


def get_tool(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def specs_for(tags: tuple[str, ...] | list[str] | None = None,
              names: list[str] | None = None) -> list[dict]:
    """按场景裁剪工具清单 —— 少给模型无关工具，既省钱又降错。

    tags 之间有交集即命中；names 则精确指定。
    """
    selected: list[Tool] = []
    for item in _REGISTRY.values():
        if names is not None:
            if item.name in names:
                selected.append(item)
            continue
        if not tags or set(item.tags) & set(tags):
            selected.append(item)
    return [t.to_llm_spec() for t in selected]


def llm_tools_spec() -> list[dict]:
    """全量工具清单（长尾自主 Agent 用）。"""
    return [t.to_llm_spec() for t in _REGISTRY.values()]


def registry_summary() -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "dangerous": t.dangerous,
            "risk_level": t.risk_level,
            "tags": list(t.tags),
            "idempotent": t.idempotent,
        }
        for t in _REGISTRY.values()
    ]


def clear_registry() -> None:
    """仅测试用。"""
    global _LOADED, _MCP_ATTACHED
    _REGISTRY.clear()
    _LOADED = False
    _MCP_ATTACHED = False


def load_all(attach_mcp: bool = True) -> None:
    """导入业务工具模块（幂等），并按配置接入 MCP Server。

    注意 clear_registry() 之后再调本函数：`from . import business` 对**已导入**的模块
    不会重新执行模块体，装饰器也就不会重新注册。所以这种情况下必须 reload，
    否则测试里"清空注册表再重新加载"会得到一个空注册表。
    """
    global _LOADED, _MCP_ATTACHED
    if not _LOADED:
        import importlib

        from . import analysis, business

        if not _REGISTRY:
            importlib.reload(business)
            importlib.reload(analysis)

        _LOADED = True

    if attach_mcp and not _MCP_ATTACHED:
        _MCP_ATTACHED = True          # 无论成功失败都只尝试一次，避免反复拉起子进程
        try:
            from ..config import settings

            if settings.mcp_servers:
                from .mcp_client import attach_configured_servers

                attach_configured_servers(verbose=True)
        except Exception as exc:  # noqa: BLE001 —— MCP 接不上不能影响本地工具可用
            print(f"[MCP] 接入跳过：{type(exc).__name__}: {exc}")
