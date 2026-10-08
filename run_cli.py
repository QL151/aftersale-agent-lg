"""命令行入口：跑工单、看路由、审关键动作、查 Bad Case。

在 aftersale-agent 目录下执行：

    python run_cli.py --doctor                            # 能力体检（哪些依赖真接了）
    python run_cli.py --tools                             # 19 个工具清单
    python run_cli.py --intents                           # 意图路由表（8 类工作流）
    python run_cli.py "帮我查下订单 SO2026000001 的物流到哪了"
    python run_cli.py "我要退款，订单号 SO2026000003"        # 会触发人工审批
    python run_cli.py --approvals                         # 看待审批的关键动作
    python run_cli.py --approve APXXXX --note "同意"        # 审批并**自动恢复**被中断的图
    python run_cli.py --bad-cases                         # 批量筛 Bad Case
    python run_cli.py --explain <run_id>                  # 单条 Trace 的因果链 + 根因
    python run_cli.py --explain                           # 同上，但自动取最近一条 Bad Case
"""

from __future__ import annotations

import argparse
import json
import sys

# 中文 Windows 控制台默认 GBK，必须显式切 UTF-8，否则打印中文会乱码/报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from app.config import settings
from app.db import Repo, reset_db
from app.hitl import ApprovalManager
from app.obs.badcase import bad_case_report, explain, latest_run_id
from app.obs.tracer import load_traces
from app.router import INTENTS, describe_routing
from app.runner import AftersaleAgent
from app.runtime import print_doctor
from app.seed import seed
from app.tools import all_tools, load_all, registry_summary

TAG = {
    "route": "[路由]", "node": "[节点]", "llm_call": "[思考]", "tool_call": "[执行]",
    "guardrail": "[拦截]", "approval": "[审批]", "final": "[结论]", "error": "[错误]",
    "cost": "[成本]",
}


def print_step(step) -> None:
    detail = step.detail
    tag = f"[{step.index:>2}] {TAG.get(step.type, '[' + step.type + ']')}"

    if step.type == "route":
        print(f"{tag} 意图={detail.get('label')}（{detail.get('confidence')}）"
              f" 来源={detail.get('source')} → "
              f"{detail.get('route')}{'/' + detail['workflow'] if detail.get('workflow') else ''}")
    elif step.type == "node":
        print(f"{tag} 节点 {detail.get('node')} 完成（{step.duration_ms}ms）")
    elif step.type == "llm_call":
        calls = detail.get("tool_calls") or []
        if calls:
            for call in calls:
                print(f"{tag} 模型决定调用：{call['name']}"
                      f"({json.dumps(call['arguments'], ensure_ascii=False)})")
        else:
            print(f"{tag} 模型给出结论（{step.duration_ms}ms，"
                  f"tokens={detail.get('prompt_tokens')}+{detail.get('completion_tokens')}）")
    elif step.type == "tool_call":
        status = "成功" if detail.get("ok") else "失败"
        print(f"{tag} {detail.get('tool')}("
              f"{json.dumps(detail.get('arguments', {}), ensure_ascii=False)}) {status}"
              f"（{step.duration_ms}ms）")
        output = str(detail.get("output") or detail.get("error") or "")
        for line in output.splitlines()[:4]:
            print(f"          | {line}")
    elif step.type == "guardrail":
        print(f"{tag} {detail.get('tool')} 被拦截 —— {detail.get('reason')}")
    elif step.type == "approval":
        print(f"{tag} {detail.get('tool')} 需人工审批 —— {detail.get('reason')}")


def cmd_tools() -> int:
    load_all()
    items = registry_summary()
    # 按"是否需要人/策略把关"分两档，而不是按数量凑一个好看的说法
    guarded = [i for i in items if i["dangerous"]]
    plain = [i for i in items if not i["dangerous"]]
    print(f"共注册 {len(items)} 个工具")
    print(f"  · 查询与流程类 {len(plain)} 个（只读，直接放行）")
    print(f"  · 需策略/人工把关 {len(guarded)} 个"
          f"（{', '.join(i['name'] for i in guarded)}）\n")

    for item in items:
        flags = []
        if item["dangerous"]:
            flags.append(f"需把关/{item['risk_level']}")
        if item["idempotent"]:
            flags.append("幂等")
        print(f"  {item['name']:<26} [{', '.join(flags) if flags else '只读'}]")
        print(f"      {item['description'][:96]}")
    print()
    print("提示：真实模型只看到 JSON Schema；工具描述是 Agent 效果的第一生产力。")
    print("      MCP 接入后这里会多出远端工具（配置 MCP_SERVERS 后重跑）。")
    return 0


def cmd_run(args) -> int:
    agent = AftersaleAgent(on_event=None, engine=args.engine)
    print("=" * 92)
    print(f"工单：{args.task}")
    print(f"模型：{'离线假模型(MockLLM)' if settings.use_mock else settings.llm_model}"
          f"｜阈值 {settings.router_confidence_threshold}"
          f"｜自动放行上限 ¥{settings.approval_auto_approve_max}"
          f"｜最多 {settings.max_steps} 步")
    print("=" * 92)

    result = agent.run(args.task, session_id=args.session)

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
    print(f"Trace：data/traces/{result.run_id}.json")

    if result.stopped_reason == "interrupted":
        print()
        print(">>> 这条工单卡在人工审批上。用下面命令继续：")
        print("    python run_cli.py --approvals")
        print(f"    python run_cli.py --approve <approval_id>   （会话 {result.thread_id}）")

    if args.json:
        print("\n完整 Trace：")
        print(result.to_json())
    return 0


def cmd_approvals() -> int:
    print(ApprovalManager().render_pending())
    return 0


def cmd_decide(approval_id: str, approve: bool, by: str, note: str) -> int:
    agent = AftersaleAgent()
    record = agent.hitl.get(approval_id)
    if not record:
        print(f"没有找到审批单 {approval_id}")
        return 1
    print(f"审批单 {approval_id}：{record['action']} 金额={record['amount']} "
          f"风险={record['risk_level']}")
    print(f"  原因：{record['reason']}")
    print(f"  会话：{record['thread_id']}")
    print(f"  决策：{'批准' if approve else '驳回'}（{by}）{note}")
    print("-" * 92)

    outcome = agent.decide_approval(approval_id, approve, by=by, note=note)
    if not outcome.get("ok"):
        print(f"失败：{outcome.get('error')}")
        return 1
    if not outcome.get("resumed"):
        print(f"审批已记录，但恢复图失败：{outcome.get('error')}")
        return 1

    result = outcome["result"]
    print("图已恢复并继续执行完成。")
    print(f"  结束原因：{result['stopped_reason']}")
    print(f"  是否全自动闭环：{result['auto_closed']}")
    print(f"  最终答复：\n{result['answer'][:600]}")
    return 0


def cmd_traces(limit: int) -> int:
    runs = load_traces(limit=limit)
    if not runs:
        print('还没有 Trace。先跑一条：python run_cli.py "帮我查下订单 SO2026000001 的物流"')
        return 0
    print(f"最近 {len(runs)} 条工单：\n")
    for run in runs:
        flag = "闭环" if run.get("auto_closed") else ("转人工" if run.get("handover") else "未闭环")
        print(f"  {run['run_id']}  {run['saved_at']}  [{flag}] {run.get('route')}"
              f"/{run.get('workflow') or '-'}")
        print(f"      意图={run.get('intent')} 工具={run.get('tool_calls')}次"
              f" 拦截={run.get('blocked_calls')}次 tokens={run.get('total_tokens')}"
              f" 成本≈¥{run.get('cost_yuan')} 耗时={run.get('total_ms')}ms")
        print(f"      工单：{run.get('task', '')[:70]}")
    print("\n看单条因果链：python run_cli.py --explain <run_id>（把 <run_id> 换成上面的编号；"
          "不想挑就直接 python run_cli.py --explain）")
    return 0


def cmd_metrics() -> int:
    with Repo() as repo:
        stats = repo.ticket_stats()
    print("业务库工单统计（含历史基线）：")
    for key, value in stats.items():
        print(f"  {key:<18}{value}")

    runs = load_traces(limit=1000)
    if runs:
        total = len(runs)
        auto = sum(1 for r in runs if r.get("auto_closed"))
        handover = sum(1 for r in runs if r.get("handover"))
        blocked = sum(1 for r in runs if r.get("blocked_calls"))
        cost = sum(float(r.get("cost_yuan") or 0) for r in runs)
        tokens = sum(int(r.get("total_tokens") or 0) for r in runs)
        print()
        print(f"Agent 处理统计（最近 {total} 条）：")
        print(f"  全自动闭环       {auto}/{total} = {auto / total * 100:.1f}%")
        print(f"  转人工           {handover}/{total} = {handover / total * 100:.1f}%")
        print(f"  触发过护栏拦截   {blocked}/{total} = {blocked / total * 100:.1f}%")
        print(f"  累计 tokens      {tokens}")
        print(f"  累计成本         ≈¥{cost:.4f}")
        print(f"  平均成本/单      ≈¥{cost / total:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="电商售后业务 Agent（工具调用 + 工作流 + 人机协同）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", nargs="?", help="工单内容，例如：我要退款，订单号 SO2026000003")
    parser.add_argument("--doctor", action="store_true", help="运行时能力体检")
    parser.add_argument("--tools", action="store_true", help="列出全部工具")
    parser.add_argument("--intents", action="store_true", help="打印意图路由表")
    parser.add_argument("--approvals", action="store_true", help="查看待人工审批的关键动作")
    parser.add_argument("--approve", metavar="APPROVAL_ID", help="批准并恢复被中断的图")
    parser.add_argument("--reject", metavar="APPROVAL_ID", help="驳回并恢复被中断的图")
    parser.add_argument("--note", default="", help="审批备注（进审计）")
    parser.add_argument("--by", default="人工客服", help="审批人")
    parser.add_argument("--traces", action="store_true", help="列出历史 Trace")
    parser.add_argument("--metrics", action="store_true", help="统计指标（闭环率/成本等）")
    parser.add_argument("--bad-cases", action="store_true", help="批量扫描 Bad Case")
    parser.add_argument("--explain", metavar="RUN_ID", nargs="?", const="last",
                        help="单条 Trace 的因果链与根因；不带值时看最近一条 Bad Case")
    parser.add_argument("--session", default="", help="会话 ID（多轮上下文）")
    parser.add_argument("--engine", default=None, choices=["native", "langgraph", "auto"],
                        help="强制状态机引擎（默认按配置自动探测）")
    parser.add_argument("--limit", type=int, default=20, help="--traces 显示条数")
    parser.add_argument("--seed", action="store_true", help="重新生成演示数据")
    parser.add_argument("--reset-db", action="store_true", help="清空并重建业务库")
    parser.add_argument("--json", action="store_true", help="输出完整 Trace JSON")
    args = parser.parse_args()

    settings.ensure_dirs()

    if args.doctor:
        print_doctor()
        return 0
    if args.reset_db:
        reset_db()
        info = seed(force=True)
        print("业务库已重建：")
        for key, value in info.items():
            print(f"  {key:<18}{value}")
        return 0
    if args.seed:
        info = seed(force=True)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    if args.tools:
        return cmd_tools()
    if args.intents:
        print(describe_routing())
        return 0
    if args.approvals:
        return cmd_approvals()
    if args.approve:
        return cmd_decide(args.approve, True, args.by, args.note)
    if args.reject:
        return cmd_decide(args.reject, False, args.by, args.note)
    if args.traces:
        return cmd_traces(args.limit)
    if args.metrics:
        return cmd_metrics()
    if args.bad_cases:
        print(bad_case_report())
        return 0
    if args.explain:
        run_id = args.explain
        if run_id.lower() in ("last", "latest"):
            run_id = latest_run_id()
            if not run_id:
                print("还没有任何 Trace。先跑一条工单：")
                print('  python run_cli.py "帮我查下订单 SO2026000001 的物流到哪了"')
                return 1
            print(f"（没指定 run_id，取最近一条：{run_id}）\n")
        print(explain(run_id))
        return 0
    if not args.task:
        parser.print_help()
        return 1

    # 第一次运行自动造数，保证 "clone 下来就能跑"
    from app.seed import ensure_seeded

    ensure_seeded()
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
