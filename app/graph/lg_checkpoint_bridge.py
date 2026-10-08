"""把本项目自定义 Checkpointer 桥接成 LangGraph 的 BaseCheckpointSaver。

为什么需要这个文件？
  项目里有自己的 Checkpointer（FileCheckpointer / RedisCheckpointer / ChainCheckpointer），
  它们提供了"状态离开进程"的能力，是 HITL 跨进程恢复的基础。
  但 LangGraph 1.x 的 `compile(checkpointer=...)` 只接受 `BaseCheckpointSaver` 实例，
  传自定义对象会直接抛：
      TypeError: Invalid checkpointer provided. Expected an instance of `BaseCheckpointSaver`,
      `True`, `False`, or `None`. Received FileCheckpointer.

  两种"修法"：
    ① 简单粗暴：`compile(checkpointer=InMemorySaver())` —— 一行改完，但**丢掉跨进程持久化**，
       重启后待审批的会话就找不回来了，HITL 卖点被打折。
    ② 桥接（本文件）：把项目的 Checkpointer 包装成合法的 BaseCheckpointSaver，
       底层存储仍然是项目的 File/Redis —— 既满足 LangGraph 接口，又保留跨进程能力。

  这里选 ②，因为它才是"LangGraph 真接入"而不是"为了跑起来把功能阉掉"。
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)


class ProjectCheckpointSaver(BaseCheckpointSaver):
    """适配器：LangGraph 的读写请求 → 转发给项目自己的 Checkpointer。

    存储布局（放在 payload 的 "__langgraph__" 键下，与项目自研引擎的
    state/next_node/pending 字段互不干扰，同一个 thread 两个引擎都能读）：
        {
          "state": {...}, "next_node": "...", "pending": ...,   # 自研引擎用
          "__langgraph__": {"checkpoint": ..., "metadata": ...} # LangGraph 用
        }
    """

    def __init__(self, store: Any, graph_name: str = "") -> None:
        super().__init__()
        self.store = store
        self.graph_name = graph_name
        self.serde = None  # 本项目状态本身就是 JSON 可序列化的普通 dict

    # ------------------------------------------------------------ 序列化
    def _load_blob(self, thread_id: str) -> dict:
        try:
            raw = self.store.load(thread_id)
        except Exception:  # noqa: BLE001 —— 存储层故障不能让图崩掉
            return {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _deserialize(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    # ------------------------------------------------------------ 读
    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not thread_id:
            return None
        blob = self._load_blob(str(thread_id))
        box = blob.get("__langgraph__")
        if not box:
            return None
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": str(thread_id),
                    "checkpoint_ns": "",
                    "checkpoint_id": box.get("checkpoint_id") or thread_id,
                }
            },
            checkpoint=self._deserialize(box["checkpoint"]),
            metadata=self._deserialize(box.get("metadata") or {}),
            parent_config=None,
            pending_writes=None,
        )

    def list(
        self, config: dict | None, *, filter: dict | None = None,
        before: dict | None = None, limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """项目场景下只按 thread 逐一恢复，不需要列历史快照。"""
        if not config:
            return
        found = self.get_tuple(config)
        if found is not None:
            yield found

    # ------------------------------------------------------------ 写
    def put(
        self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> dict:
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not thread_id:
            return config
        thread_id = str(thread_id)

        blob = self._load_blob(thread_id)
        cp_id = checkpoint.get("id") or thread_id
        blob["__langgraph__"] = {
            "checkpoint": json.loads(json.dumps(checkpoint, default=str)),
            "metadata": json.loads(json.dumps(metadata or {}, default=str)),
            "checkpoint_id": cp_id,
        }
        # 关键：同时在 payload 里写一份项目格式的 "state"。
        #   ① 与自研引擎的存档格式保持一致，同一个 thread 两个引擎都能读；
        #   ② 跨进程 resume 时，适配器靠它把真实 AgentState 读回来
        #      （否则被重跑的中断节点拿不到状态对象）。
        values = (checkpoint.get("channel_values") or {})
        agent_state = values.get("agent_state")
        if agent_state is not None and hasattr(agent_state, "to_payload"):
            blob["state"] = agent_state.to_payload()
        elif isinstance(agent_state, dict) and agent_state.get("run_id"):
            blob["state"] = agent_state

        # 注意：不要在这里写 blob["graph"] = "langgraph"！
        # "graph" 字段是**工作流名**（如 refund_request），恢复时 runner 靠它
        # 重新编译对应的图。LangGraph 的 checkpoint 里没有这个信息，
        # 必须由适配器把图名传进来（graph_name）。
        if self.graph_name and not blob.get("graph"):
            blob["graph"] = self.graph_name
        blob.setdefault("next_node", blob.get("next_node", ""))
        try:
            self.store.save(thread_id, blob)
        except Exception:  # noqa: BLE001
            pass

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": cp_id,
            }
        }

    def put_writes(self, config: dict, writes, task_id: str, task_path: str = "") -> None:  # noqa: ARG002
        """中间写入（如 Command 的 update）暂不需要单独落库：
        最终 checkpoint 里已包含完整状态，恢复语义不受影响。"""
        return None

    # ------------------------------------------------------------ 命名
    @property
    def name(self) -> str:
        return f"project-bridge({getattr(self.store, 'name', 'store')})"
