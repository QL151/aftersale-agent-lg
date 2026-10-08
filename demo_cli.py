"""交互式命令行演示。

跟 run_cli.py 的区别：
  run_cli.py     一次跑一条，工单靠命令行参数传入（适合脚本/自动化）
  demo_cli.py    一个小循环，输入即执行（适合坐下来连续试几条）

为什么不直接在 .bat 里读输入：cmd 会把工单里的 & | > < 当成语法符号，
「退款金额>200」这种文本会被截断甚至执行意外命令。交给 Python 读输入就没这问题。
"""

from __future__ import annotations

import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from app.config import settings          # noqa: E402
from app.hitl import ApprovalManager     # noqa: E402
from app.runner import AftersaleAgent    # noqa: E402
from run_cli import print_step           # noqa: E402


def run_one(task: str, session: str = "") -> None:
    agent = AftersaleAgent(on_event=None)
    print("=" * 92)
    print(f"工单：{task}")
    print(f"模型：{'离线假模型(MockLLM)' if settings.use_mock else settings.llm_model}"
          f"｜意图阈值 {settings.router_confidence_threshold}"
          f"｜自动放行上限 ¥{settings.approval_auto_approve_max}")
    print("=" * 92)

    result = agent.run(task, session_id=session)

    print("-" * 92)
    for step in result.steps:
        print_step(step)
    print("-" * 92)
    print(f"结论：\n{result.answer}")
    print("-" * 92)
    print(f"路由={result.route}/{result.workflow or '-'}｜结束={result.stopped_reason}"
          f"｜工具 {result.tool_calls} 次/拦截 {result.blocked_calls} 次"
          f"｜tokens={result.total_tokens}｜成本≈¥{result.cost_yuan:.4f}"
          f"｜引擎={result.engine}")
    print(f"全自动闭环={result.auto_closed}｜转人工={result.handover}"
          f"｜人工审批={result.approvals}")

    if result.stopped_reason == "interrupted":
        print()
        print(">>> 这条工单卡在人工审批上。输入 a 看待审批，输入 y <审批号> 批准。")
    print()


def decide_one(approval_id: str, approve: bool = True) -> None:
    agent = AftersaleAgent()
    record = agent.hitl.get(approval_id)
    if not record:
        print(f"没有找到审批单 {approval_id}\n")
        return
    print(f"审批单 {approval_id}：{record['action']} 金额={record['amount']} "
          f"风险={record['risk_level']}")
    print(f"  原因：{record['reason']}")

    outcome = agent.decide_approval(approval_id, approve, by="人工客服", note="命令行操作")
    if not outcome.get("ok"):
        print(f"失败：{outcome.get('error')}\n")
        return
    if not outcome.get("resumed"):
        print(f"审批已记录，但恢复图失败：{outcome.get('error')}\n")
        return

    result = outcome["result"]
    print("图已恢复并继续执行完成。")
    print(f"  结束原因：{result['stopped_reason']}｜是否全自动闭环：{result['auto_closed']}")
    print(f"  最终答复：\n{result['answer'][:600]}")
    print()


def show_map() -> None:
    """复用项目里的数据地图脚本（列出按场景可用的订单号）。"""
    subprocess.run([sys.executable, os.path.join(BASE, "data_map.py")], check=False)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if "--map" in sys.argv:
        show_map()
        return 0

    print("=" * 92)
    print("  电商售后业务 Agent  -  交互式演示")
    print("=" * 92)
    print("  直接输入工单内容，回车即执行：")
    print("    帮我查一下订单 SO2026000001 的物流到哪了")
    print("    帮我退一下订单 SO2026000013 的款，商品有质量问题")
    print()
    print("  d           查看可用订单（数据地图：哪些订单号能用）")
    print("  a           查看待人工审批")
    print("  y <审批号>   批准并恢复被中断的工单")
    print("  q           退出")
    print("=" * 92)
    print()

    while True:
        try:
            raw = input("工单 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        low = raw.lower()
        if low in ("q", "quit", "exit"):
            break
        if low == "d":
            show_map()
            print()
            continue
        if low == "a":
            print(ApprovalManager().render_pending())
            print()
            continue
        if low.startswith("y "):
            decide_one(raw[2:].strip(), approve=True)
            continue

        try:
            run_one(raw)
        except Exception as exc:                      # noqa: BLE001
            print(f"[错误] {type(exc).__name__}: {exc}\n")

    print("已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
