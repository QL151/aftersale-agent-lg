"""沙箱执行工具：让 Agent 能对售后数据做真实的数据分析，而不是靠嘴编。

典型场景：
  「最近 30 天各品类的退款率是多少」「哪几家承运商异常率最高」
  —— 这类问题用工具算比让模型硬猜可靠得多，也正好演示"代码执行型 Agent"。

关键设计：**模型只能写代码，不能写 SQL，也拿不到数据库连接**。
  工具内部把数据取成 JSON 列表塞进沙箱（叫 DATA），
  模型在沙箱里只能算这份内存数据。这样即使代码有问题，也碰不到生产库。
"""

from __future__ import annotations

from ..sandbox import Sandbox, build_sandbox
from .base import ToolContext, tool

_DATASET_SQL = {
    "orders": (
        "SELECT o.order_id, o.category, o.sku_name, o.amount, o.shipping_fee, o.status,"
        " o.paid_at, o.delivered_at, o.is_opened, c.level AS customer_level"
        " FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id"
    ),
    "tickets": (
        "SELECT ticket_id, intent, status, assignee, auto_closed, channel,"
        " baseline_minutes, handled_minutes, created_at FROM tickets"
    ),
    "logistics": (
        "SELECT s.tracking_no, s.carrier, s.status, o.category, o.order_id"
        " FROM shipments s LEFT JOIN orders o ON o.order_id = s.order_id"
    ),
}

_SANDBOX: Sandbox | None = None


def _sandbox() -> Sandbox:
    """沙箱实例复用（避免每次调用都探测一次 Docker）。"""
    global _SANDBOX
    if _SANDBOX is None:
        _SANDBOX = Sandbox(build_sandbox())
    return _SANDBOX


@tool(
    name="run_analysis_code",
    description=(
        "在**受限沙箱**里执行一段 Python 代码，对售后数据做统计分析。"
        "当问题涉及比例、分布、趋势、Top-N、相关性等需要真实计算时使用它，不要靠估计。\n"
        "代码约定：数据已经在变量 `DATA` 里（list[dict]）；用 print() 输出结论，"
        "或把结构化结果赋给 `RESULT`。\n"
        "可用模块：math/statistics/json/re/datetime/collections/itertools/functools/decimal。\n"
        "禁止：文件读写、网络、导入 os/sys/subprocess、eval/exec、访问 dunder 属性。\n"
        "示例：\n"
        "  from collections import Counter\n"
        "  c = Counter(r['category'] for r in DATA)\n"
        "  print('各品类订单量：', c.most_common())\n"
        "  RESULT = dict(c)"
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "minLength": 3, "maxLength": 6000, "description": "要执行的 Python 代码"},
            "dataset": {
                "type": "string",
                "enum": ["orders", "tickets", "logistics"],
                "description": "数据集：orders=订单（含客户等级）、tickets=历史工单、logistics=物流运单",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    dangerous=True,          # 执行代码，交 Guardrail 走 sandbox 策略
    risk_level="medium",
    policy="sandbox",        # ← 内容级策略：AST 校验通过才放行（不需要人工确认）
    timeout=20.0,
    tags=("analysis",),
    max_output=4000,
)
def run_analysis_code(
    code: str, dataset: str = "orders", _ctx: ToolContext | None = None
) -> str:
    sql = _DATASET_SQL.get(dataset)
    if sql is None:
        raise ValueError(f"未知数据集 {dataset}（可选：{list(_DATASET_SQL)}）")

    if _ctx is not None and _ctx.repo is not None:
        rows = _ctx.repo.query(sql)
    else:
        from ..db import Repo

        with Repo() as repo:
            rows = repo.query(sql)

    result = _sandbox().run(code, rows)
    header = f"[沙箱后端 {result.backend}｜数据集 {dataset}｜{len(rows)} 行｜{result.duration_ms}ms]"
    return header + "\n" + result.to_text()
