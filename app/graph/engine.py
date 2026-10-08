"""引擎选择：同一张图，编译到自研引擎或真实 LangGraph。

调用方（工作流 / Agent）只写一遍图，这里决定它跑在哪里。
"""

from __future__ import annotations

from typing import Any, Callable

from ..config import settings
from ..runtime import langgraph_available
from . import core
from .checkpoint import build_checkpointer


def build_engine(
    graph: core.StateGraph,
    checkpointer: Any = None,
    recursion_limit: int | None = None,
    max_seconds: float | None = None,
    on_node: Callable[[str, Any, int], None] | None = None,
    force: str | None = None,
) -> Any:
    """把声明式图编译成可执行图。

    force / settings.graph_engine：auto | langgraph | native
      · langgraph 显式要求但没装 → 回退 native 并**打印警告**（不静默降级）
      · 这样面试演示时你能说清"当前跑的是哪个引擎、为什么"
    """
    engine = (force or settings.graph_engine or "auto").lower()
    limit = recursion_limit or settings.max_steps
    seconds = max_seconds if max_seconds is not None else settings.max_total_seconds
    cp = checkpointer if checkpointer is not None else build_checkpointer()

    if engine in ("langgraph", "auto") and langgraph_available():
        try:
            from .langgraph_adapter import compile_langgraph

            # on_node 必须一起传下去：控制台/SSE 的"每个节点 + 每次工具调用"
            # 全靠它上报。早先这里漏传，导致装了 langgraph 以后
            # 左边「执行过程」只剩「路由」和「完成」两行，看不到中间步骤。
            return compile_langgraph(graph, cp, limit, seconds, on_node=on_node)
        except Exception as exc:  # noqa: BLE001 —— 适配失败不能拖垮业务
            print(f"[警告] LangGraph 编译失败，回退自研引擎：{type(exc).__name__}: {exc}")
    elif engine == "langgraph":
        print(
            "[警告] 配置要求使用 LangGraph，但当前环境未安装（pip install langgraph）。"
            "已回退到自研等价引擎 —— 语义一致，可正常跑通。"
        )

    return graph.compile(
        checkpointer=cp, recursion_limit=limit, max_seconds=seconds, on_node=on_node
    )
