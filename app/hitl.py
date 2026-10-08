"""人机协同（HITL）：关键动作必须有人类点头。

哪些动作要人？不是"所有危险动作"，而是**不可逆 + 有成本**的动作：
  退款（动钱）→ 必须审批
  发券（有成本）→ 必须审批
  查订单（只读）→ 不用，否则客服会被确认框淹没

三层设计（这才是能上线的形态）：
  ① **小额免打扰**：金额 ≤ 阈值且风险不高 → 自动放行，不打断人
  ② **中断-恢复**：图中断（interrupt）→ 状态落盘 → 人工决定 → 恢复（resume）
     中间进程可以重启，因为决策的唯一真相在数据库里
  ③ **超时兜底**：超时未处理 → 按配置转人工工单或拒绝（绝不会"忘了审批就一直挂在那"）

关键工程细节（面试可以重点讲）：
  · approval_id 是**确定性推导**的（run_id + 动作 + 参数），
    所以图 resume 时节点重跑不会产生重复审批单。
  · 恢复时以**数据库里的决策为准**，而不只信 resume 传入的值 ——
    这样"API 先落决策、再唤醒图"和"进程重启后再唤醒"都能正确工作。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from . import clock
from .config import settings
from .db import Repo
from .graph.core import interrupt

# 风险等级 → 是否必须人工
MUST_APPROVE = {"high"}
AUTO_OK = {"low"}


@dataclass
class ApprovalDecision:
    approved: bool
    by: str = ""
    note: str = ""
    approval_id: str = ""
    auto: bool = False
    timed_out: bool = False
    reason: str = ""


def make_approval_id(run_id: str, action: str, arguments: dict) -> str:
    """确定性审批单号 —— 让"节点重跑"天然幂等。"""
    raw = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{run_id}|{action}|{raw}".encode("utf-8")).hexdigest()[:12].upper()
    return f"AP{digest}"


class ApprovalManager:
    def __init__(
        self,
        repo: Repo | None = None,
        timeout_seconds: int | None = None,
        timeout_action: str | None = None,
        auto_approve_max: float | None = None,
    ) -> None:
        self._repo = repo
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.approval_timeout_seconds
        )
        self.timeout_action = timeout_action or settings.approval_timeout_action
        self.auto_approve_max = (
            auto_approve_max if auto_approve_max is not None else settings.approval_auto_approve_max
        )

    @property
    def repo(self) -> Repo:
        if self._repo is None:
            self._repo = Repo()
        return self._repo

    # ------------------------------------------------------------ 查询
    def get(self, approval_id: str) -> dict[str, Any] | None:
        return self.repo.get_approval(approval_id)

    def pending(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.repo.list_approvals(status="pending", limit=limit)

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.repo.list_approvals(status="", limit=limit)

    # ------------------------------------------------------------ 决策
    def decide(
        self, approval_id: str, approved: bool, by: str = "人工客服", note: str = "",
        timed_out: bool = False,
    ) -> bool:
        """写入人工决策。返回是否成功（重复决策返回 False）。"""
        status = "timeout" if timed_out else ("approved" if approved else "rejected")
        return self.repo.decide_approval(approval_id, status, by, note)

    def sweep_timeouts(self) -> int:
        """超时兜底：把挂太久的审批单按策略收掉。"""
        settled = 0
        for item in self.pending(limit=200):
            created = clock.parse(item["created_at"])
            if created is None:
                continue
            age = (clock.now() - created).total_seconds()
            if age <= self.timeout_seconds:
                continue
            approved = self.timeout_action == "approve"
            self.decide(
                item["approval_id"], approved, by="系统(超时)",
                note=f"超过 {self.timeout_seconds}s 未处理，按策略 {self.timeout_action}",
                timed_out=True,
            )
            settled += 1
        return settled

    # ------------------------------------------------------------ 图内使用
    def gate(
        self,
        state: Any,
        *,
        action: str,
        arguments: dict[str, Any],
        reason: str,
        risk_level: str = "high",
        amount: float = 0.0,
        force: bool = False,
    ) -> dict[str, Any]:
        """在图的节点里调用：需要审批就挂起，不需要就直接放行。

        force=True 表示"调用方已判定必须人工"（例如工作流里金额超阈值），
        此时即使金额很小也不做小额免打扰，但仍然把真实金额记进审批单供审核人判断。

        返回值会被合并进 state：
          {"pending_approval": {...}, "approval_denied": bool, "approvals": int}
        """
        run_id = getattr(state, "run_id", "")
        thread_id = getattr(state, "thread_id", "")

        # ① 小额免打扰
        if not force and risk_level not in MUST_APPROVE and amount and amount <= self.auto_approve_max:
            return {
                "approvals": getattr(state, "approvals", 0) + 1,
                "approval_denied": False,
                "pending_approval": None,
                "facts": {
                    **getattr(state, "facts", {}),
                    "_approval": {
                        "approved": True, "auto": True, "by": "系统(小额免打扰)",
                        "approval_id": "", "note": f"金额 {amount} ≤ 阈值 {self.auto_approve_max}",
                    },
                },
            }

        approval_id = make_approval_id(run_id, action, arguments)
        self.repo.create_approval(
            approval_id=approval_id, run_id=run_id, thread_id=thread_id, action=action,
            arguments=json.dumps(arguments, ensure_ascii=False), reason=reason,
            risk_level=risk_level, amount=amount,
        )

        payload = {
            "approval_id": approval_id,
            "action": action,
            "arguments": arguments,
            "reason": reason,
            "risk_level": risk_level,
            "amount": amount,
            "run_id": run_id,
            "thread_id": thread_id,
            "timeout_seconds": self.timeout_seconds,
            "timeout_action": self.timeout_action,
        }

        # ② 挂起等人工。resume 时节点从头重跑，这里会直接拿到唤醒值。
        wake_value = interrupt(payload)

        # ③ 以数据库为准：兼容"API 先落库再唤醒"和"进程重启后唤醒"
        record = self.get(approval_id) or {}
        status = record.get("status", "pending")
        if status == "pending" and isinstance(wake_value, dict):
            status = "approved" if wake_value.get("approved") else "rejected"
            self.decide(approval_id, bool(wake_value.get("approved")),
                        by=wake_value.get("by", "人工客服"), note=wake_value.get("note", ""))
            record = self.get(approval_id) or {}
            status = record.get("status", status)

        approved = status == "approved"
        decision = {
            "approved": approved,
            "auto": False,
            "by": record.get("decided_by", ""),
            "note": record.get("decision_note", ""),
            "approval_id": approval_id,
            "timed_out": status == "timeout",
            "status": status,
        }
        return {
            "approvals": getattr(state, "approvals", 0) + 1,
            "approval_denied": not approved,
            "pending_approval": None,
            "facts": {**getattr(state, "facts", {}), "_approval": decision},
        }

    # ------------------------------------------------------------ 展示
    @staticmethod
    def render_pending() -> str:
        manager = ApprovalManager()
        items = manager.pending()
        if not items:
            return "当前没有待审批的关键动作。"
        lines = [f"待审批 {len(items)} 条（退款/发券等关键动作）：", ""]
        for item in items:
            lines.append(
                f"  {item['approval_id']}  [{item['risk_level']:<6}] "
                f"{item['action']}({item['arguments']})  金额={item['amount']}"
            )
            lines.append(f"      原因：{item['reason']}")
            lines.append(f"      提交于 {item['created_at']}  会话={item['thread_id']}")
        lines.append("")
        lines.append("审批：POST /approvals/{approval_id}/decide  （或 CLI --explain 查看上下文）")
        return "\n".join(lines)
