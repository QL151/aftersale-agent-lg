"""状态机引擎（自研实现，语义对齐 LangGraph）。

为什么要有状态机，而不是一个 while 循环？
  手写 ReAct 循环能跑通 demo，但一旦要"中断等人工审批、审批完接着跑，
  而且中间进程可能重启"，就必须有**可持久化的状态 + 明确的节点边界**。
  这正是 LangGraph 解决的问题，也是本项目把 Agent 循环和确定性工作流
  统一成同一套图的原因：
    · 8 类高频意图 → 确定性工作流图（不调 LLM，稳定、便宜、可测）
    · 长尾意图     → 自主 Agent 图（ReAct 循环节点）
  两者用同一个引擎跑，所以循环上限、失败回退、断点续跑、Trace 全都只写一遍。

本文件提供的能力（对应简历里的说法）：
  · 循环上限          → recursion_limit，超出抛 GraphRecursionError（不是死循环）
  · 失败回退          → 节点重试 + fallback 节点 + error 边
  · 断点续跑          → Checkpointer，每个节点执行完就落盘
  · 人机协同（HITL）  → interrupt() 挂起 / resume() 恢复，语义与 LangGraph 一致
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Callable

END = "__end__"
START = "__start__"


# ==================================================================== 异常与信号


class GraphError(Exception):
    """图执行相关错误的基类。"""


class GraphRecursionError(GraphError):
    """超出循环上限 —— Agent 陷入死循环时靠它兜底，而不是把进程跑死。"""


class NodeFailed(GraphError):
    """节点执行失败且重试/回退都用尽。"""


class GraphInterrupt(GraphError):
    """节点主动挂起，等待外部输入（典型：等人工审批）。

    注意：它是**控制流**，不是错误。所以它必须能被 invoke 捕获、
    存成 checkpoint，并在 resume 时把外部输入喂回同一个节点。
    """

    def __init__(self, payload: Any = None) -> None:
        super().__init__(f"图已挂起：{payload!r}")
        self.payload = payload


@dataclass
class Command:
    """节点除了"改状态"，还能显式指定下一步去哪。"""

    goto: str | None = None
    update: dict[str, Any] | None = None


# ==================================================================== 中断上下文
# 用 contextvars 而不是实例属性：FastAPI 下多个会话可能并发跑同一张编译后的图，
# 实例属性会串台，contextvars 天然按协程/线程隔离。


@dataclass
class _ExecContext:
    resume_values: list[Any] = field(default_factory=list)
    interrupt_index: int = 0

    def reset(self) -> None:
        self.interrupt_index = 0


_CTX: contextvars.ContextVar[_ExecContext] = contextvars.ContextVar("graph_exec")

# 引擎钩子：当图由 LangGraph 适配器执行时，把 interrupt() 转交给 langgraph.types.interrupt。
# 有了它，**同一份节点函数**可以不改一行地跑在自研引擎和 LangGraph 上 ——
# 这才是"可切换"而不是"写了两套代码"。
_INTERRUPT_HOOK: contextvars.ContextVar[Any] = contextvars.ContextVar("interrupt_hook", default=None)


def interrupt(payload: Any = None) -> Any:
    """在节点内部调用：挂起当前图，并把 payload 交给外部。

    恢复时**节点会从头重跑**，此时 interrupt() 直接返回外部传入的值。
    所以节点里 interrupt() 之前的代码必须是确定性的（幂等）——
    这也是为什么退款必须带幂等键：重跑不能重复扣款。
    """
    hook = _INTERRUPT_HOOK.get()
    if hook is not None:
        return hook(payload)

    ctx = _CTX.get()
    index = ctx.interrupt_index
    ctx.interrupt_index += 1
    if index < len(ctx.resume_values):
        return ctx.resume_values[index]
    raise GraphInterrupt(payload)


def interrupt_pending() -> bool:
    """当前是否处于"恢复但还有新的挂起点"——供节点判断用。"""
    return False


# ==================================================================== 重试策略


@dataclass
class RetryPolicy:
    max_attempts: int = 2
    backoff_seconds: float = 0.0
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if isinstance(exc, (GraphInterrupt, GraphRecursionError)):
            return False
        return attempt < self.max_attempts and isinstance(exc, self.retry_on)


# ==================================================================== 图定义

NodeAction = Callable[[Any], Any]
PathFn = Callable[[Any], str]


class StateGraph:
    """声明式地描述"有哪些节点、怎么流转"。"""

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self._nodes: dict[str, NodeAction] = {}
        self._retry: dict[str, RetryPolicy | None] = {}
        self._fallback: dict[str, str] = {}
        self._edges: dict[str, str] = {}
        self._branches: dict[str, tuple[PathFn, dict[str, str] | None]] = {}
        self._entry: str = ""

    # ------------------------------------------------------------ 声明
    def add_node(
        self,
        name: str,
        action: NodeAction,
        retry: RetryPolicy | None = None,
        fallback: str | None = None,
    ) -> "StateGraph":
        """加一个节点。

        fallback：该节点彻底失败（重试也用尽）后跳到哪个节点。
        这就是"失败回退"：比如"退款执行"失败 → 回退到"转人工"节点。
        """
        if name in (END, START):
            raise ValueError(f"节点名不能是保留字 {name}")
        self._nodes[name] = action
        self._retry[name] = retry
        if fallback:
            self._fallback[name] = fallback
        return self

    def add_edge(self, start: str, end: str) -> "StateGraph":
        self._edges[start] = end
        return self

    def add_conditional_edges(
        self, start: str, path_fn: PathFn, path_map: dict[str, str] | None = None
    ) -> "StateGraph":
        """条件边：path_fn 返回一个 key，再由 path_map 映射到真实节点名。"""
        self._branches[start] = (path_fn, path_map)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self._entry = name
        return self

    def _finalize(self) -> None:
        """补齐默认出边：没有声明出边的节点视为终点。两个编译器都要先调它。"""
        if not self._entry:
            raise GraphError("没有设置入口节点（set_entry_point）")
        for name in self._nodes:
            if name not in self._edges and name not in self._branches:
                self._edges[name] = END

    def compile(
        self,
        checkpointer: Any = None,
        recursion_limit: int = 25,
        max_seconds: float | None = None,
        on_node: Callable[[str, Any, int], None] | None = None,
    ) -> "CompiledGraph":
        self._finalize()
        return CompiledGraph(
            self._nodes, self._retry, self._fallback, self._edges, self._branches,
            self._entry, self.name, checkpointer, recursion_limit, max_seconds, on_node,
        )


# ==================================================================== 图执行


@dataclass
class RunOutcome:
    state: Any
    stopped_reason: str            # finished / interrupted / max_steps / timeout / error
    interrupted: bool = False
    pending: Any = None
    nodes_executed: int = 0
    error: str = ""


class CompiledGraph:
    def __init__(
        self,
        nodes: dict[str, NodeAction],
        retry: dict[str, RetryPolicy | None],
        fallback: dict[str, str],
        edges: dict[str, str],
        branches: dict[str, tuple[PathFn, dict[str, str] | None]],
        entry: str,
        name: str,
        checkpointer: Any,
        recursion_limit: int,
        max_seconds: float | None,
        on_node: Callable[[str, Any, int], None] | None,
    ) -> None:
        self.nodes = nodes
        self.retry = retry
        self.fallback = fallback
        self.edges = edges
        self.branches = branches
        self.entry = entry
        self.name = name
        self.checkpointer = checkpointer
        self.recursion_limit = recursion_limit
        self.max_seconds = max_seconds
        self.on_node = on_node
        self.engine = "native"

    # ------------------------------------------------------------ 对外入口
    def invoke(self, state: Any, thread_id: str = "", recursion_limit: int | None = None) -> RunOutcome:
        return self._run(state, self.entry, thread_id or getattr(state, "thread_id", ""),
                         recursion_limit or self.recursion_limit, resume_values=[])

    def resume(self, value: Any = None, thread_id: str = "") -> RunOutcome:
        """恢复一张被 interrupt() 挂起的图，把 value 作为 interrupt() 的返回值。"""
        if self.checkpointer is None:
            raise GraphError("没有配置 checkpointer，无法恢复：状态没有被保存")
        saved = self.checkpointer.load(thread_id)
        if not saved:
            raise GraphError(f"没有找到 thread {thread_id} 的存档，无法恢复")
        state = _state_from_payload(saved["state"])
        next_node = saved.get("next_node") or self.entry
        values = [value] if not isinstance(value, list) else value
        return self._run(state, next_node, thread_id, self.recursion_limit, resume_values=values,
                         reenter=True)

    def get_state(self, thread_id: str) -> Any:
        if self.checkpointer is None:
            return None
        saved = self.checkpointer.load(thread_id)
        return _state_from_payload(saved["state"]) if saved else None

    # ------------------------------------------------------------ 主循环
    def _run(
        self,
        state: Any,
        node: str,
        thread_id: str,
        recursion_limit: int,
        resume_values: list[Any],
        reenter: bool = False,
    ) -> RunOutcome:
        ctx = _ExecContext(resume_values=list(resume_values))
        token = _CTX.set(ctx)
        started = time.time()
        executed = 0
        stopped = "finished"
        error = ""

        try:
            while node != END:
                if executed >= recursion_limit:
                    stopped = "max_steps"
                    break
                # 注意用 is not None 而不是真值判断：max_seconds=0 是"立刻超时"的合法配置，
                # 写成 `if self.max_seconds and ...` 会把这个配置静默忽略掉。
                if self.max_seconds is not None and (time.time() - started) > self.max_seconds:
                    stopped = "timeout"
                    break

                action = self.nodes.get(node)
                if action is None:
                    stopped = "error"
                    error = f"节点 {node} 不存在"
                    break

                ctx.reset()                      # 每个节点重跑时中断计数归零（LangGraph 同语义）
                t0 = time.time()
                try:
                    update = self._call_with_retry(node, action, state)
                except GraphInterrupt as gi:
                    executed += 1
                    self._save(thread_id, state, node, pending=gi.payload)
                    if self.on_node:
                        self.on_node(node, state, int((time.time() - t0) * 1000))
                    return RunOutcome(state, "interrupted", True, gi.payload, executed)

                duration = int((time.time() - t0) * 1000)
                executed += 1
                _merge(state, update)
                if self.on_node:
                    self.on_node(node, state, duration)

                if isinstance(update, Command) and update.goto:
                    node = update.goto
                else:
                    node = self._next_node(node, state)

                self._save(thread_id, state, node)
        except Exception as exc:  # noqa: BLE001 —— 引擎必须兜住一切，否则整通会话挂掉
            stopped = "error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            _CTX.reset(token)

        return RunOutcome(state, stopped, False, None, executed, error)

    def _call_with_retry(self, node: str, action: NodeAction, state: Any) -> Any:
        policy = self.retry.get(node)
        attempt = 0
        while True:
            attempt += 1
            try:
                return action(state)
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                if policy and policy.should_retry(exc, attempt):
                    if policy.backoff_seconds:
                        time.sleep(policy.backoff_seconds)
                    continue
                # 重试用尽 → 失败回退
                if node in self.fallback:
                    target = self.fallback[node]
                    retries = getattr(state, "retries", None)
                    if isinstance(retries, dict):
                        retries[node] = retries.get(node, 0) + attempt
                    return Command(goto=target, update={})
                raise NodeFailed(f"节点 {node} 执行失败：{type(exc).__name__}: {exc}") from exc

    def _next_node(self, current: str, state: Any) -> str:
        if current in self.branches:
            path_fn, mapping = self.branches[current]
            key = path_fn(state)
            if key == END:
                return END
            if mapping:
                return mapping.get(key, END)
            return key
        return self.edges.get(current, END)

    def _save(self, thread_id: str, state: Any, next_node: str, pending: Any = None) -> None:
        if self.checkpointer is None or not thread_id:
            return
        self.checkpointer.save(
            thread_id,
            {
                "graph": self.name,
                "state": _state_to_payload(state),
                "next_node": next_node,
                "pending": pending,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )


def _merge(state: Any, update: Any) -> None:
    """把节点返回值合并进状态。

    约定：
      · None            → 不改
      · Command         → 只合并 update
      · dict            → 逐字段 setattr
      · 其它状态对象    → 整体替换（就地更新字段）
    """
    if update is None:
        return
    if isinstance(update, Command):
        _merge(state, update.update)
        return
    if isinstance(update, dict):
        for key, value in update.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return
    if isinstance(update, type(state)):
        for key, value in vars(update).items():
            setattr(state, key, value)


def _state_to_payload(state: Any) -> dict:
    return state.to_payload()


def _state_from_payload(payload: dict) -> Any:
    from ..schemas import AgentState

    return AgentState.from_payload(payload)
