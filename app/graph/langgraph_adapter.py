"""LangGraph 适配器：把本项目声明的图，编译成真正的 LangGraph 图。

为什么要有这一层？
  招聘 JD 里写的是 LangGraph，本项目也**确实接入了 LangGraph**（不是假装）。
  但本项目同时需要一个"零依赖也能跑"的等价引擎，否则任何人在没有中间件、
  没有外网的环境下都跑不起来 —— 那就没法演示、没法让人复现。

  所以做法是：**图只声明一次**（app/graph/core.py 的 StateGraph），
  然后有两个编译器：
    · compile_native()     —— 自研引擎，默认，零依赖
    · compile_langgraph()  —— 编译成 langgraph.graph.StateGraph，切换真实 LangGraph

  两边语义对齐的四个点，是这个适配器唯一需要翻译的东西：
    1. 状态      AgentState(dataclass)  ↔  dict 状态（放在 "agent_state" 键下）
    2. 中断      core.interrupt()       ↔  langgraph.types.interrupt()
                 通过 core._INTERRUPT_HOOK 注入，节点函数零改动
    3. 跳转      core.Command(goto=..)  ↔  langgraph.types.Command(goto=..)
    4. 存档      本项目 Checkpointer     ↔  LangGraph Checkpointer
                 通过 ProjectCheckpointSaver 桥接（见 lg_checkpoint_bridge.py），
                 底层仍是项目的 File/Redis 存储 —— 跨进程恢复能力不丢。

已在 langgraph 1.2.12 / langgraph-checkpoint-sqlite 3.1.1 上实测通过：
  · `python -m app.doctor`            → 引擎显示 langgraph
  · `python run_cli.py --engine langgraph "..."` → 物流查询跑通
  · 退款审批 interrupt → 进程重启 → resume → 退款落库（跨进程恢复）
"""

from __future__ import annotations


import time
from typing import Annotated, Any, Callable, TypedDict

from . import core


class LangGraphUnavailable(RuntimeError):
    pass


# 哨兵：区分"没有中断"和"中断了但载荷为 None"——两者语义完全不同，
# 用 None 当哨兵会把"中断且无载荷"误判成"已结束"。
_NO_INTERRUPT = object()

# 合法状态类型，用于节点重跑时识别"传进来的是状态还是 interrupt 返回值"。
from ..schemas import AgentState as _AgentState  # noqa: E402

_STATE_TYPES: tuple[type, ...] = (_AgentState,)


# ---------------------------------------------------------------- 状态 schema
# 只声明一个通道 agent_state，并配"整体替换"reducer。
# 为什么不直接 LGStateGraph(dict)？
#   裸 dict 会被 LangGraph 当作 TypedDict 处理，把里面所有键展开成顶层通道，
#   于是一个 dataclass 状态就被肢解成 N 个字段，收敛时会走到
#   AgentState(**payload) —— 而 AgentState 有必填字段，直接 TypeError。
#   声明单通道 + 替换 reducer，则状态永远是"一个整体对象"，与自研引擎语义一致。
#
# 注意：必须定义在**模块级**。LangGraph 用 typing.get_type_hints() 解析注解，
# 它只认模块全局命名空间，函数内定义的 TypedDict 会因为找不到 Annotated 而报错。
def _replace(_old: Any, new: Any) -> Any:
    """整体替换，不做字段合并 —— 保证状态对象不被肢解。"""
    return new


class GraphStateDict(TypedDict):
    agent_state: Annotated[Any, _replace]


def _import_langgraph():
    try:
        # 1.x 里叫 InMemorySaver；旧版叫 MemorySaver。两个都试，避免版本差异。
        try:
            from langgraph.checkpoint.memory import InMemorySaver as _MemorySaver
        except ImportError:  # pragma: no cover - 旧版兼容
            from langgraph.checkpoint.memory import MemorySaver as _MemorySaver
        from langgraph.checkpoint.base import BaseCheckpointSaver
        from langgraph.graph import END as LG_END
        from langgraph.graph import StateGraph as LGStateGraph
        from langgraph.types import Command as LGCommand
        from langgraph.types import interrupt as lg_interrupt
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise LangGraphUnavailable(
            f"未安装 langgraph（{exc}）。执行 `pip install -r requirements.txt` 后可用；"
            "当前请使用自研引擎（--engine native）。"
        ) from exc
    return LGStateGraph, LG_END, LGCommand, lg_interrupt, _MemorySaver, BaseCheckpointSaver


def _coerce_saver(checkpointer: Any, MemorySaver, BaseCheckpointSaver,
                  graph_name: str = "") -> Any:
    """把项目自定义 Checkpointer 适配成 LangGraph 认可的 saver。

    LangGraph 1.x 的 compile() 只接受 BaseCheckpointSaver 实例，传项目自己的
    FileCheckpointer / ChainCheckpointer 会直接 TypeError。这里做一次桥接：
      · 已经是 BaseCheckpointSaver  → 原样返回
      · None                        → InMemorySaver（进程内，够测试用）
      · 项目自定义 Checkpointer      → ProjectCheckpointSaver 包一层，
                                      底层持久化能力（文件/Redis）保留
    graph_name 会随存档落盘，跨进程恢复时用来重新编译对应的图。
    """
    if checkpointer is None:
        return MemorySaver()
    if isinstance(checkpointer, BaseCheckpointSaver):
        return checkpointer
    # 项目的 Checkpointer：包成合法 saver，跨进程恢复能力不丢
    from .lg_checkpoint_bridge import ProjectCheckpointSaver

    return ProjectCheckpointSaver(checkpointer, graph_name=graph_name)


class LangGraphCompiled:
    """用 LangGraph 执行，但对外暴露与自研引擎相同的接口（invoke / resume / get_state）。"""

    engine = "langgraph"

    def __init__(self, graph: core.StateGraph, checkpointer: Any, recursion_limit: int,
                 max_seconds: float | None,
                 on_node: Callable[[str, Any, int], None] | None = None) -> None:
        LGStateGraph, LG_END, LGCommand, lg_interrupt, MemorySaver, BaseSaver = _import_langgraph()

        self._LGCommand = LGCommand
        self._lg_interrupt = lg_interrupt
        self.recursion_limit = recursion_limit
        self.max_seconds = max_seconds
        # 节点执行完回调一次（与自研引擎 core.py 的 on_node 同语义）：
        # runner 靠它把「节点 + 该节点内的工具调用/拦截/审批」推给前端。
        self.on_node = on_node
        # 图名要落进 checkpoint：跨进程恢复时 runner 靠它重新编译对应的图。
        # 自研引擎在自己的 _save 里写这个字段，LangGraph 路径必须补上，
        # 否则恢复时会报"存档里的图 已不存在"。
        self.graph_name = graph.name
        # 唯一的兼容处理：自定义 Checkpointer → LangGraph 认可的 BaseCheckpointSaver
        self.saver = _coerce_saver(checkpointer, MemorySaver, BaseSaver,
                                   graph_name=self.graph_name)
        # 最近的合法状态对象：节点重跑时如果拿到的是 interrupt 返回值，用它兜底
        self._last_state: Any = None
        # 节点名 → 它本次要用 Command 去的目标（由出边路由消费，见 _edge_router）
        self._next_hop: dict[str, str] = {}

        graph._finalize()          # 与自研引擎共用同一套"补齐默认边"逻辑

        builder = LGStateGraph(GraphStateDict)
        for name, action in graph._nodes.items():
            builder.add_node(name, self._wrap(name, action, LGCommand, lg_interrupt))
        builder.set_entry_point(graph._entry)

        # 边 / 条件边
        #
        # ⚠️ 这里有个 LangGraph 与自研引擎的**语义差异**，是本适配器最关键的一处翻译：
        #   自研引擎里 Command(goto=X) **优先于**该节点的静态出边（core.py:318）；
        #   但 LangGraph 里 Command(goto=X) 与静态边是**两套并行机制**——
        #   节点既声明了静态边、又返回 Command(goto=) 时，LangGraph 会
        #   **两边都调度**。实测后果非常严重：
        #     退款审批被驳回时返回 Command(goto="denied")，
        #     静态边 approval→execute 也会被调度 → **驳回后居然照常退款**。
        #
        # 修法：**项目里一律不向 LangGraph 暴露 Command**。
        #   节点执行时若返回了 Command，我们把它翻译成"状态里的一个路由标记"
        #   （_next_hop），然后把该节点的所有出边统一注册为**条件边**：
        #     路由函数先看 _next_hop 有没有值 —— 有就直接去那里（动态优先）；
        #     没有才回落到原本的静态目标。
        #   这样 LangGraph 里只有"条件边"一套路由机制，不再存在双调度。
        for start, end in graph._edges.items():
            target = LG_END if end == core.END else end
            # 不传 path_map：路由函数可能返回"静态目标"或"Command 的动态目标"，
            # 两者都是真实节点名/END 常量。传了 path_map 反而会把动态目标判成非法。
            builder.add_conditional_edges(start, self._edge_router(start, target))
        for start, (path_fn, mapping) in graph._branches.items():
            resolved = None
            if mapping:
                resolved = {
                    key: (LG_END if value == core.END else value)
                    for key, value in mapping.items()
                }
                if core.END in resolved:
                    pass
                else:
                    resolved[core.END] = LG_END
            builder.add_conditional_edges(start, self._wrap_router(path_fn), resolved)

        self.compiled = builder.compile(checkpointer=self.saver)

    # ------------------------------------------------------------ 出边路由
    def _edge_router(self, node_name: str, fallback_target: str):
        """统一出边路由：Command 的动态目标优先，否则回落静态目标。

        项目节点的返回类型有两种：
          · dict / None      → 走图的静态出边（fallback_target）
          · Command(goto=X)  → 走 X（自研引擎里 Command 优先，这里对齐）
        节点包装阶段会把 Command 的目标写进 self._next_hop[node_name]，
        路由函数据此决定去向 —— 从而彻底避开 LangGraph 的双调度问题。
        """

        def pick(state: dict) -> str:
            hop = getattr(self, "_next_hop", {}).get(node_name)
            if hop:
                return LG_END if hop == core.END else hop
            return fallback_target

        return pick

    # ------------------------------------------------------------ 条件边包装
    @staticmethod
    def _wrap_router(path_fn):
        """条件边也要做参数适配。

        项目的 path_fn 签名是 `(AgentState) -> str`，但 LangGraph 传给它的是
        **图的顶层状态 dict**（{"agent_state": ...}）。不做适配就会走到
        LangGraph 的 _coerce_state → schema(**input) → AgentState(**dict)，
        而 AgentState 有必填字段，直接 TypeError。节点做了包装，条件边同样要包。
        """

        def router(state: dict) -> str:
            agent_state = state.get("agent_state") if isinstance(state, dict) else state
            return path_fn(agent_state)

        return router

    # ------------------------------------------------------------ 节点包装
    def _wrap(self, node_name, action, LGCommand, lg_interrupt):
        def node(state: dict) -> dict:
            agent_state = state.get("agent_state") if isinstance(state, dict) else state

            # LangGraph 有个必须绕开的语义陷阱：
            #   resume 时它会**重跑**被中断的节点，并把 interrupt() 的返回值
            #   当作该节点对 agent_state 通道的输出。由于我们用的是"整体替换"
            #   reducer，真实状态对象会被这个返回值覆盖掉（实测变成 'yes' 字符串）。
            #   所以这里做一次守卫：只要拿到的不是合法状态对象，就用
            #   self._last_state（invoke/resume 时记下的真实状态）兜底。
            if not isinstance(agent_state, _STATE_TYPES):
                agent_state = self._last_state
            if agent_state is None:
                raise RuntimeError("LangGraph 节点拿不到有效的 AgentState（状态已丢失）")
            self._last_state = agent_state

            # 让 core.interrupt() 走 LangGraph 的中断机制（节点函数零改动）
            hook_token = core._INTERRUPT_HOOK.set(lg_interrupt)
            started = time.time()
            try:
                update = action(agent_state)
            except BaseException:
                # 节点里 interrupt() 时 LangGraph 用异常表达挂起，异常会越过下面的
                # 上报点。这里补一次，否则控制台会漏掉「提交审批」那一步。
                self._report_node(node_name, agent_state, started)
                raise
            finally:
                core._INTERRUPT_HOOK.reset(hook_token)

            # 与自研引擎共用同一套合并语义（none/dict/Command/状态对象）。
            # 节点普遍写 `return {"answer": ...}`，漏掉这一步 answer 就永远写不回状态，
            # 表现为"跑通了但没结论"。
            core._merge(agent_state, update)

            if isinstance(update, core.Command):
                # 不向 LangGraph 暴露 Command（会与静态边双调度，导致走错路径）。
                # 改为把目标记进 _next_hop，由 _edge_router 在读边时消费。
                self._next_hop[node_name] = update.goto or core.END
            elif node_name in self._next_hop:
                # 节点重跑后这次没返回 Command，旧标记要清掉，避免误路由
                self._next_hop.pop(node_name, None)

            self._report_node(node_name, agent_state, started)
            return {"agent_state": agent_state}

        return node

    def _report_node(self, node_name: str, state: Any, started: float) -> None:
        """向上层上报"某节点执行完毕"（含耗时）。

        时机与自研引擎一致：正常路径在状态合并之后上报；
        中断路径没有合并，就在抛出前上报。
        上报本身出错绝不能影响业务（回调里会写 trace、推 SSE），所以整段兜住。
        """
        if self.on_node is None:
            return
        try:
            self.on_node(node_name, state, int((time.time() - started) * 1000))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ 对外接口
    def invoke(self, state: Any, thread_id: str = "", recursion_limit: int | None = None) -> core.RunOutcome:
        tid = thread_id or getattr(state, "thread_id", "")
        config = {
            "configurable": {"thread_id": tid},
            "recursion_limit": recursion_limit or self.recursion_limit,
        }
        self._last_state = state        # 记为兜底状态，供重跑的中断节点使用
        self._next_hop.clear()          # 每次新执行重置路由标记
        try:
            result = self.compiled.invoke({"agent_state": state}, config)
        except Exception as exc:  # noqa: BLE001
            return self._handle_exception(state, exc)

        # LangGraph 1.x 的 interrupt() 不会把异常冒泡给 invoke：
        # 它在内部被 Pregel 捕获，以 `__interrupt__` 字段出现在返回值里，
        # 同时 snapshot.next 指向"待重跑的节点"。所以这里必须显式检查，
        # 否则中断会被误判成 finished（表现为"该等审批却直接跑完了"）。
        pending = self._extract_interrupt(result, config)
        if pending is not _NO_INTERRUPT:
            return core.RunOutcome(state, "interrupted", True, pending, 0)
        return core.RunOutcome(state, "finished", False, None, 0)

    @staticmethod
    def _interrupt_value(raw: Any) -> Any:
        """把 LangGraph 的 Interrupt 对象拆成裸载荷。"""
        return getattr(raw, "value", raw)

    def _extract_interrupt(self, result: Any, config: dict) -> Any:
        """从 invoke 返回值 / 快照里取出中断载荷。没中断返回 _NO_INTERRUPT 哨兵。"""
        if isinstance(result, dict):
            raw = result.get("__interrupt__")
            if raw:
                first = raw[0] if isinstance(raw, (list, tuple)) else raw
                return self._interrupt_value(first)
        try:
            snapshot = self.compiled.get_state(config)
        except Exception:  # noqa: BLE001
            return _NO_INTERRUPT
        if snapshot is not None and getattr(snapshot, "next", None):
            # 有待执行节点 = 被挂起。快照里可能带 interrupt 载荷，取不到就置 None。
            tasks = getattr(snapshot, "tasks", None) or ()
            for task in tasks:
                interrupts = getattr(task, "interrupts", None) or ()
                if interrupts:
                    return self._interrupt_value(interrupts[0])
            return None
        return _NO_INTERRUPT

    def resume(self, value: Any = None, thread_id: str = "") -> core.RunOutcome:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}

        # 跨进程恢复的关键：新进程里 self._last_state 是空的，
        # 必须先从 checkpointer 把上次落盘的真实状态读回来，
        # 否则被重跑的中断节点会拿不到 AgentState（表现为 AttributeError）。
        self._last_state = self._load_saved_state(thread_id)
        self._next_hop.clear()          # 恢复也要重置路由标记

        try:
            result = self.compiled.invoke(self._LGCommand(resume=value), config)
        except Exception as exc:  # noqa: BLE001
            return self._handle_exception(None, exc)

        # 恢复后可能又碰到新的 interrupt（一条流程里可以有多个审批点），
        # 所以 resume 也要走同一套中断检测，不能一律当 finished。
        pending = self._extract_interrupt(result, config)
        if pending is not _NO_INTERRUPT:
            agent_state = (result or {}).get("agent_state") if isinstance(result, dict) else None
            return core.RunOutcome(agent_state, "interrupted", True, pending, 0)

        values = (result or {}).get("agent_state") if isinstance(result, dict) else None
        return core.RunOutcome(values, "finished", False, None, 0)

    def _load_saved_state(self, thread_id: str) -> Any:
        """从 checkpointer 里把上次落盘的真实 AgentState 读回来（跨进程恢复用）。

        桥接器 ProjectCheckpointSaver 的底层是项目的 File/Redis 存储，
        payload 里除了 LangGraph 自己的 __langgraph__ 区块，还有自研引擎写入的
        "state"（to_payload 结果）—— 直接复用它，不用去解析 LangGraph 的 checkpoint。
        """
        if not thread_id:
            return None
        try:
            raw = self.saver.store.load(thread_id)
        except Exception:  # noqa: BLE001
            return None
        payload = (raw or {}).get("state")
        if not payload:
            return None
        try:
            return _AgentState.from_payload(payload)
        except Exception:  # noqa: BLE001
            return None

    def get_state(self, thread_id: str) -> Any:
        # 优先从项目存储读（字段完整、可跨引擎共用）；
        # 读不到再退回 LangGraph 快照里的 agent_state。
        saved = self._load_saved_state(thread_id)
        if saved is not None:
            return saved
        snapshot = self.compiled.get_state({"configurable": {"thread_id": thread_id}})
        values = getattr(snapshot, "values", None) or {}
        return values.get("agent_state")

    def _handle_exception(self, state: Any, exc: BaseException) -> core.RunOutcome:
        """把 LangGraph 的异常翻译成与自研引擎一致的 RunOutcome。"""
        name = type(exc).__name__
        if name == "GraphRecursionError":
            return core.RunOutcome(state, "max_steps", False, None, 0)
        # LangGraph 在节点里 interrupt() 时，通过 GraphInterrupt 表达挂起
        if name == "GraphInterrupt" or "interrupt" in name.lower():
            payload = getattr(exc, "args", (None,))
            pending = payload[0] if payload else None
            if isinstance(pending, (list, tuple)) and pending:
                pending = getattr(pending[0], "value", pending[0])
            return core.RunOutcome(state, "interrupted", True, pending, 0)
        return core.RunOutcome(state, "error", False, None, 0, f"{name}: {exc}")


def compile_langgraph(
    graph: core.StateGraph, checkpointer: Any = None, recursion_limit: int = 25,
    max_seconds: float | None = None,
    on_node: Callable[[str, Any, int], None] | None = None,
) -> LangGraphCompiled:
    return LangGraphCompiled(graph, checkpointer, recursion_limit, max_seconds, on_node=on_node)
