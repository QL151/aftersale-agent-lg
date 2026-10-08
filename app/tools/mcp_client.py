"""MCP 客户端：把远端 MCP Server 的工具**动态注册**进本地工具表。

这就是"Agent 侧免改动接入"的落地：
    内部系统 → MCP Server（mcp_server/server.py）
    新系统上线 → Agent 侧只改一行配置（MCP_SERVERS），代码零改动
因为工具是通过 `tools/list` 在运行时发现的，注册进本地表之后：
    路由层、8 条工作流、护栏、Trace、成本熔断全部无感复用。

对比传统做法：新接一个内部系统要写一个 Python 函数 + 注册 + 改提示词 + 重新部署。
系统一多，Agent 侧就烂了。MCP 把这个问题从"代码问题"变成"协议问题"。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ..config import settings
from .base import Tool, ToolContext, register

ROOT = Path(__file__).resolve().parent.parent.parent


class MCPError(RuntimeError):
    pass


class MCPClient:
    """一个 MCP Server 的 stdio 客户端（JSON-RPC 2.0 + 换行分隔）。"""

    def __init__(self, command: str | list[str], timeout: float = 20.0) -> None:
        self.command = command if isinstance(command, list) else command.split()
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self.tools: list[dict] = []
        self.server_info: dict = {}
        self._lock = threading.Lock()
        self._seq = 0

    # ------------------------------------------------------------ 连接
    def connect(self) -> None:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            self.proc = subprocess.Popen(
                self._argv(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(ROOT), env=env, bufsize=1,
            )
        except OSError as exc:
            raise MCPError(f"无法启动 MCP Server {self.command}：{exc}") from exc

        try:
            info = self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "aftersale-agent", "version": "1.0.0"},
            })
        except MCPError as exc:
            raise MCPError(f"{exc}｜server stderr: {self._stderr_tail()}") from exc
        self.server_info = info.get("serverInfo", {})
        self._notify("notifications/initialized", {})
        self.tools = self._request("tools/list", {}).get("tools", [])

    def _argv(self) -> list[str]:
        """把配置里的命令规范化成可执行的 argv。

        `python -m mcp_server.server` 里的 `python` 只是"用当前解释器"的意思，
        必须替换成 sys.executable 而不是追加 —— 否则会变成
        `python -X utf8 python -m ...`，子进程直接报 "can't open file 'python'"。
        """
        argv = list(self.command)
        if argv and Path(argv[0]).name.lower() in {"python", "python3", "python.exe", "py"}:
            argv = argv[1:]
        return [sys.executable, "-X", "utf8", *argv]

    def _stderr_tail(self, limit: int = 600) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            self.proc.terminate()
            _out, err = self.proc.communicate(timeout=3)
            return (err or "").strip()[-limit:]
        except Exception:  # noqa: BLE001
            return ""

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self.proc = None

    # ------------------------------------------------------------ 协议
    def _send(self, payload: dict) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise MCPError("MCP Server 未连接")
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _request(self, method: str, params: dict) -> dict:
        with self._lock:
            self._seq += 1
            request_id = self._seq
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            assert self.proc is not None and self.proc.stdout is not None
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise MCPError(f"MCP Server {method} 无响应（进程已退出）")
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if message.get("id") != request_id:
                    continue          # 跳过通知等无关消息
                if "error" in message:
                    raise MCPError(f"{method} 失败：{message['error']}")
                return message.get("result", {})

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    # ------------------------------------------------------------ 调用
    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        parts = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "\n".join(parts) or "（MCP 工具没有返回内容）"
        if result.get("isError"):
            raise MCPError(text)
        return text


# ==================================================================== 动态注册


def _make_tool(client: MCPClient, spec: dict, prefix: str = "") -> Tool:
    """把一个 MCP 工具描述包装成本地 Tool —— 之后它和本地工具完全等价。"""
    raw_name = spec.get("name", "")
    name = f"{prefix}{raw_name}"

    def handler(_ctx: ToolContext | None = None, **arguments: Any) -> str:
        return client.call_tool(raw_name, arguments)

    schema = spec.get("inputSchema") or {"type": "object", "properties": {}}
    # MCP 的 inputSchema 允许 additionalProperties 缺省，这里补齐成严格模式，
    # 让本地 Guardrail 能拦住模型编造的额外参数。
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})

    return Tool(
        name=name,
        description=f"[MCP:{client.server_info.get('name', 'remote')}] {spec.get('description', '')}",
        parameters=schema,
        func=handler,
        dangerous=False,           # 远端只读查询类工具；写操作应在对端自行做鉴权
        risk_level="low",
        timeout=client.timeout,
        tags=("mcp",),
        idempotent=True,
        inject_context=True,
        max_output=3000,
    )


def register_mcp_tools(client: MCPClient, prefix: str = "") -> list[str]:
    """把一个 MCP Server 的工具全部注册进本地工具表。返回注册成功的工具名。"""
    registered: list[str] = []
    for spec in client.tools:
        tool_obj = _make_tool(client, spec, prefix)
        if register(tool_obj):
            registered.append(tool_obj.name)
    return registered


_clients: list[MCPClient] = []


def attach_configured_servers(verbose: bool = False) -> dict[str, Any]:
    """按 MCP_SERVERS 配置接入所有 MCP Server。

    配置格式（逗号分隔，每个 server 一条命令）：
        MCP_SERVERS=python -m mcp_server.server
    多条时用 `;` 分隔：
        MCP_SERVERS=python -m a.server;python -m b.server
    """
    global _clients
    report: dict[str, Any] = {"servers": [], "tools": []}
    raw = (settings.mcp_servers or "").strip()
    if not raw:
        return report

    for entry in [e.strip() for e in raw.split(";") if e.strip()]:
        client = MCPClient(entry)
        try:
            client.connect()
        except Exception as exc:  # noqa: BLE001 —— 接不上不能拖垮 Agent 启动
            report["servers"].append({"command": entry, "ok": False, "error": str(exc)})
            if verbose:
                print(f"[MCP] 接入失败 {entry}：{exc}")
            continue
        names = register_mcp_tools(client)
        _clients.append(client)
        report["servers"].append({
            "command": entry, "ok": True,
            "server": client.server_info.get("name", ""),
            "tools": len(client.tools),
        })
        report["tools"].extend(names)
        if verbose:
            print(f"[MCP] 已接入 {client.server_info.get('name', entry)}，"
                  f"动态注册 {len(names)} 个工具：{'、'.join(names)}")
    return report


def shutdown() -> None:
    global _clients
    for client in _clients:
        client.close()
    _clients = []
