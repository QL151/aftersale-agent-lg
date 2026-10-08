"""MCP Server：把内部订单 / CRM / 物流系统封装成标准 MCP 工具。

为什么这是"加分项"而不是"多写一个文件"？
  Agent 要接一个新系统，传统做法是：写一个函数 → 注册工具 → 改 Agent 的工具清单 → 重新部署。
  系统一多，Agent 侧就成了意大利面。
  MCP（Model Context Protocol）把它变成**协议问题**：
    内部系统暴露一个 MCP Server → Agent 侧只维护一个 MCP 客户端 →
    新系统上线时 Agent **一行代码都不用改**（list_tools 自动发现）。

  这就是本项目 app/tools/mcp_client.py 里 `register_mcp_tools()` 干的事：
  它把远端 MCP Server 的工具**动态注册进本地工具注册表**，
  于是路由层、工作流、护栏、Trace 全部无感复用。

实现说明（没有 mcp SDK 也能跑）：
  MCP 的 stdio 传输其实就是 **JSON-RPC 2.0 + 换行分隔**，
  所以这里用标准库手写了服务端，支持 initialize / tools/list / tools/call / ping。
  装上官方 SDK（pip install mcp）后，也可以用它把本文件包装成等价的 server —— 见文件末尾。

启动方式（单机自测）：
    python -m mcp_server.server
配置给 Agent（自动接入所有工具）：
    MCP_SERVERS=python -m mcp_server.server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 允许直接 python mcp_server/server.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "aftersale-internal-mcp", "version": "1.0.0"}

# ==================================================================== 工具定义
# 注意：这里的 inputSchema 就是 MCP 的 JSON Schema，跟 OpenAI Function Calling 的形状一致，
# 所以"从 MCP 发现工具"到"喂给模型"之间几乎零转换 —— 这是 MCP 设计得好的地方。

TOOLS: list[dict[str, Any]] = [
    {
        "name": "crm_get_customer",
        "description": "【内部 CRM】按客户号或手机号查询客户档案：等级、注册时间、累计消费、标签。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "客户号，例如 C00001"},
                "phone": {"type": "string", "description": "手机号（与客户号二选一）"},
            },
        },
    },
    {
        "name": "crm_list_customer_tags",
        "description": "【内部 CRM】查询客户标签与风险标记（用于判断是否需要人工复核）。",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string", "description": "客户号"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "oms_get_order",
        "description": "【内部订单系统 OMS】按订单号查询订单主数据与状态。",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "oms_list_customer_orders",
        "description": "【内部订单系统 OMS】列出某客户最近的订单。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "客户号"},
                "limit": {"type": "integer", "description": "返回条数，默认 5"},
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "tms_get_tracking",
        "description": "【内部物流系统 TMS】按订单号或运单号查询物流轨迹与异常标记。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
                "tracking_no": {"type": "string", "description": "运单号"},
            },
        },
    },
    {
        "name": "tms_list_abnormal",
        "description": "【内部物流系统 TMS】列出当前物流异常的运单（供批量排查用）。",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "返回条数，默认 5"}},
        },
    },
]


# ==================================================================== 工具实现


def _repo():
    from app.db import Repo

    return Repo()


def call_crm_get_customer(customer_id: str = "", phone: str = "") -> str:
    with _repo() as repo:
        customer = repo.get_customer(customer_id) if customer_id else None
        if customer is None and phone:
            rows = repo.query("SELECT customer_id FROM customers WHERE phone = ?", (phone,))
            if rows:
                customer = repo.get_customer(rows[0]["customer_id"])
        if customer is None:
            return "CRM：没有查询到该客户。"
        return (
            f"CRM 客户档案\n"
            f"  客户号：{customer.customer_id}\n"
            f"  姓名：{customer.name}\n"
            f"  等级：{customer.level}\n"
            f"  注册时间：{customer.registered_at}\n"
            f"  近 90 天退款：{customer.refund_count_90d} 次｜投诉：{customer.complaint_count_90d} 次\n"
            f"  风险评分：{customer.risk_score}"
        )


def call_crm_list_customer_tags(customer_id: str) -> str:
    with _repo() as repo:
        customer = repo.get_customer(customer_id)
        if customer is None:
            return "CRM：没有查询到该客户。"
        prefs = repo.query(
            "SELECT pref_key, pref_value FROM customer_preferences WHERE customer_id = ?",
            (customer_id,),
        )
        facts = repo.query(
            "SELECT fact FROM customer_facts WHERE customer_id = ? ORDER BY id DESC LIMIT 5",
            (customer_id,),
        )
    tags = [f"{p['pref_key']}={p['pref_value']}" for p in prefs]
    if customer.risk_score >= 0.5:
        tags.append("风险标记:需人工复核")
    if customer.level in {"金牌", "黑名单观察"}:
        tags.append(f"等级标记:{customer.level}")
    lines = [f"CRM 客户标签（{customer_id}）", "  标签：" + ("、".join(tags) if tags else "（无）")]
    if facts:
        lines.append("  近期记录：")
        lines += [f"    · {f['fact']}" for f in facts]
    return "\n".join(lines)


def call_oms_get_order(order_id: str) -> str:
    with _repo() as repo:
        order = repo.get_order(order_id)
    return f"OMS：没有找到订单 {order_id}。" if order is None else "OMS 订单主数据\n" + "\n".join(
        f"  {k}：{v}" for k, v in order.public().items()
    )


def call_oms_list_customer_orders(customer_id: str, limit: int = 5) -> str:
    with _repo() as repo:
        orders = repo.list_orders_by_customer(customer_id, limit=limit)
    if not orders:
        return f"OMS：客户 {customer_id} 名下没有订单。"
    lines = [f"OMS 客户 {customer_id} 最近 {len(orders)} 单："]
    lines += [
        f"  {o.order_id} | {o.sku_name} | {o.amount} 元 | {o.status} | 签收 {o.delivered_at or '未签收'}"
        for o in orders
    ]
    return "\n".join(lines)


def call_tms_get_tracking(order_id: str = "", tracking_no: str = "") -> str:
    with _repo() as repo:
        shipment = (
            repo.get_shipment_by_order(order_id) if order_id else repo.get_shipment_by_tracking(tracking_no)
        )
    if shipment is None:
        return f"TMS：没有查到 {order_id or tracking_no} 的物流信息（可能尚未发货）。"
    lines = [
        "TMS 物流轨迹",
        f"  运单号：{shipment.tracking_no}｜承运商：{shipment.carrier}",
        f"  当前状态：{shipment.status}",
    ]
    lines += [f"  {e.time} | {e.status} | {e.location} | {e.description}" for e in shipment.events]
    if shipment.status == "异常":
        lines.append("  ⚠ 该运单已被 TMS 标记为异常，超过 72 小时无更新。")
    return "\n".join(lines)


def call_tms_list_abnormal(limit: int = 5) -> str:
    with _repo() as repo:
        items = repo.abnormal_shipments(limit=limit)
    if not items:
        return "TMS：当前没有异常运单。"
    lines = [f"TMS 异常运单（{len(items)} 条）："]
    for shipment in items:
        last = shipment.last_event
        lines.append(
            f"  {shipment.tracking_no} | {shipment.carrier} | 订单 {shipment.order_id}"
            f" | 最后更新 {last.time if last else '-'}"
        )
    return "\n".join(lines)


HANDLERS = {
    "crm_get_customer": call_crm_get_customer,
    "crm_list_customer_tags": call_crm_list_customer_tags,
    "oms_get_order": call_oms_get_order,
    "oms_list_customer_orders": call_oms_list_customer_orders,
    "tms_get_tracking": call_tms_get_tracking,
    "tms_list_abnormal": call_tms_list_abnormal,
}


# ==================================================================== JSON-RPC


def handle(request: dict) -> dict | None:
    """处理一条 JSON-RPC 请求。返回 None 表示这是 notification（无需响应）。"""
    method = request.get("method", "")
    request_id = request.get("id")
    params = request.get("params") or {}

    if method == "initialize":
        return _ok(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })

    if method in ("notifications/initialized", "initialized"):
        return None                       # 通知，不响应

    if method == "ping":
        return _ok(request_id, {})

    if method == "tools/list":
        return _ok(request_id, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if handler is None:
            return _ok(request_id, {
                "content": [{"type": "text", "text": f"未知工具：{name}"}],
                "isError": True,
            })
        try:
            text = handler(**arguments)
            return _ok(request_id, {"content": [{"type": "text", "text": text}], "isError": False})
        except TypeError as exc:
            return _ok(request_id, {
                "content": [{"type": "text", "text": f"参数不匹配：{exc}"}], "isError": True,
            })
        except Exception as exc:  # noqa: BLE001 —— MCP 要求错误也用正常响应返回
            return _ok(request_id, {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            })

    if method in ("resources/list", "prompts/list"):
        # 本项目只暴露 tools 能力；其它能力显式返回空，避免客户端报错
        key = method.split("/")[0]
        return _ok(request_id, {key: []})

    return {
        "jsonrpc": "2.0", "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _ok(request_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(stdin=None, stdout=None) -> None:
    """换行分隔的 JSON-RPC over stdio —— 这就是 MCP 的 stdio 传输。"""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    # 先确保数据库有数据，否则 MCP 工具会返回空
    try:
        from app.seed import ensure_seeded

        ensure_seeded()
    except Exception:  # noqa: BLE001
        pass

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }, ensure_ascii=False) + "\n")
            stdout.flush()
            continue

        response = handle(request)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
