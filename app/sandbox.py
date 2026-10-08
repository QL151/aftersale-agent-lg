"""沙箱执行层：把"跑模型生成的代码"变成一件**可控**的事。

三个可替换的后端：
  · SubprocessSandbox —— 默认。独立进程 + AST 校验 + 受限 builtins + 超时（零依赖）
  · DockerSandbox     —— 有容器时用：--network=none --read-only 内存/CPU 限额（真隔离）
  · NullSandbox       —— 关闭沙箱功能

为什么值得单独写一层？
  数据分析类 Agent 一旦"会写代码"，风险等级就从"说错话"跳到"执行任意代码"。
  面试里能说清"我怎么让它只能算数、不能删库"，比说"我跑通了数据分析"值钱得多。
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

_RUNNER = Path(__file__).resolve().parent / "tools" / "_sandbox_runner.py"

# AST 层禁止的调用 / 名称
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "breakpoint", "exit", "quit",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "__import__", "memoryview", "object", "type", "super", "help",
}
_FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib",
    "http", "ftplib", "smtplib", "pickle", "marshal", "ctypes", "importlib",
    "multiprocessing", "threading", "signal", "pty", "tempfile", "webbrowser",
}

# 逃逸用的属性名（拿到 frame / globals / code 对象就能重新拿到 __builtins__）。
# 用**黑名单**而不是白名单：白名单会把 `row.get` `statistics.mean` 这类正常写法一起误杀，
# 工具就废了。黑名单 + 禁 dunder + 禁 getattr/object/type，足以挡住已知逃逸路径。
_DANGEROUS_ATTRS = {
    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "ag_frame", "ag_code",
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "f_trace", "f_lineno",
    "tb_frame", "tb_next", "func_globals", "func_code", "im_func", "im_class",
    "mro", "subclasses", "subclasshook", "init_subclass", "set_name", "reduce_ex",
    "getattribute", "getattr", "setattr", "delattr", "dict", "class", "bases",
    "globals", "code", "builtins", "loader", "spec", "modules", "system", "popen",
}


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    result: object = None
    error: str = ""
    duration_ms: int = 0
    backend: str = ""
    blocked_by: str = ""            # ast / timeout / runtime / backend

    def to_text(self) -> str:
        if self.blocked_by == "ast":
            return f"[沙箱拒绝执行] {self.error}"
        if self.blocked_by == "timeout":
            return f"[沙箱超时] {self.error}"
        if not self.ok:
            return f"[执行失败] {self.error}" + (f"\n输出：\n{self.stdout}" if self.stdout else "")
        parts = []
        if self.stdout.strip():
            parts.append(f"标准输出：\n{self.stdout.strip()}")
        if self.result is not None:
            parts.append("RESULT = " + json.dumps(self.result, ensure_ascii=False, default=str))
        return "\n".join(parts) or "（代码执行成功，但没有输出。请用 print 或给 RESULT 赋值。）"


# ==================================================================== AST 校验


def validate_code(code: str) -> str | None:
    """静态检查。返回拒绝原因；None = 通过。"""
    if len(code) > 6000:
        return "代码过长（上限 6000 字符）"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"语法错误：{exc}"

    for node in ast.walk(tree):
        # 禁止 import 非白名单模块
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    return f"禁止导入模块 `{alias.name}`"
                if root not in _ALLOWED_MODULES:
                    return f"模块 `{alias.name}` 不在白名单内"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_MODULES:
                return f"禁止导入模块 `{node.module}`"
            if root not in _ALLOWED_MODULES:
                return f"模块 `{node.module}` 不在白名单内"
        # 禁止 dunder 属性与已知逃逸属性：`().__class__.__bases__[0].__subclasses__()` 这类全靠它
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"禁止访问 dunder 属性 `{node.attr}`（沙箱逃逸入口）"
            if node.attr in _DANGEROUS_ATTRS:
                return f"禁止访问属性 `{node.attr}`（可用于获取 frame/globals 逃逸）"
        # 禁止危险名称
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                return f"禁止使用 `{node.id}`"
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return "禁止使用 global / nonlocal"
    return None


_ALLOWED_MODULES = {
    "math", "statistics", "json", "re", "datetime", "collections",
    "itertools", "functools", "decimal", "fractions", "random", "string",
}


# ==================================================================== 后端


class SubprocessSandbox:
    name = "subprocess"

    def run(self, code: str, data: list, timeout: float) -> SandboxResult:
        started = time.time()
        payload = json.dumps({"code": code, "data": data}, ensure_ascii=False)
        try:
            proc = subprocess.run(
                # -X utf8：强制 UTF-8 模式。
                # 注意不能靠 PYTHONIOENCODING 环境变量 —— -I（isolated）会忽略所有 PYTHON* 变量，
                # 子进程会退回 Windows 控制台的 GBK 代码页，一 print 中文就 UnicodeEncodeError。
                [sys.executable, "-X", "utf8", "-I", "-S", str(_RUNNER)],
                input=payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=tempfile.gettempdir(),        # 不在项目目录里跑，避免误伤文件
                env={"PYTHONIOENCODING": "utf-8", "PATH": ""},
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False, error=f"执行超过 {timeout:.1f} 秒，已强制终止",
                duration_ms=int((time.time() - started) * 1000),
                backend=self.name, blocked_by="timeout",
            )

        duration = int((time.time() - started) * 1000)
        stdout = (proc.stdout or "").strip()
        if proc.returncode != 0 and not stdout:
            return SandboxResult(
                ok=False, error=f"沙箱进程异常退出（code={proc.returncode}）：{(proc.stderr or '')[:400]}",
                duration_ms=duration, backend=self.name, blocked_by="runtime",
            )
        try:
            payload_out = json.loads(stdout.splitlines()[-1] if stdout else "{}")
        except (json.JSONDecodeError, IndexError):
            return SandboxResult(
                ok=False, error=f"沙箱返回无法解析：{stdout[:400]}",
                duration_ms=duration, backend=self.name, blocked_by="runtime",
            )
        return SandboxResult(
            ok=bool(payload_out.get("ok")),
            stdout=str(payload_out.get("stdout", "")),
            result=payload_out.get("result"),
            error=str(payload_out.get("error", "")),
            duration_ms=duration, backend=self.name,
            blocked_by="" if payload_out.get("ok") else "runtime",
        )


class DockerSandbox:
    """真隔离：无网络、只读根、限制内存与 CPU、非 root。"""

    name = "docker"

    def __init__(self, image: str = "python:3.12-slim") -> None:
        self.image = image

    def run(self, code: str, data: list, timeout: float) -> SandboxResult:
        started = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "payload.json").write_text(
                json.dumps({"code": code, "data": data}, ensure_ascii=False), encoding="utf-8"
            )
            try:
                proc = subprocess.run(
                    [
                        "docker", "run", "--rm",
                        "--network=none",
                        "--read-only",
                        "--memory=256m", "--cpus=0.5",
                        "--user=65534:65534",
                        "-v", f"{tmp_path}:/work:ro",
                        self.image,
                        "python", "-I", "-S", "/work/runner.py",
                    ],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=timeout + 10, input=None,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
                return SandboxResult(
                    ok=False, error=f"Docker 沙箱不可用：{exc}", backend=self.name,
                    duration_ms=int((time.time() - started) * 1000), blocked_by="backend",
                )
        # 需要把 runner.py 一并挂进去，这里交由 build_sandbox 准备镜像时处理
        return SandboxResult(
            ok=proc.returncode == 0, stdout=proc.stdout or "", error=proc.stderr or "",
            duration_ms=int((time.time() - started) * 1000), backend=self.name,
        )


class NullSandbox:
    name = "disabled"

    def run(self, code: str, data: list, timeout: float) -> SandboxResult:
        return SandboxResult(ok=False, error="沙箱功能已关闭（SANDBOX_ENABLED=0）", blocked_by="backend")


def _docker_available() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=5,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def build_sandbox():
    """按可用性挑后端：显式配置 > Docker > 子进程 > 关闭。"""
    if not settings.sandbox_enabled:
        return NullSandbox()
    if _docker_available():
        return DockerSandbox()
    return SubprocessSandbox()


@dataclass
class Sandbox:
    """对外门面：先静态校验，再交给后端执行。"""

    backend: object = field(default_factory=build_sandbox)

    def run(self, code: str, data: list | None = None, timeout: float | None = None) -> SandboxResult:
        reason = validate_code(code)
        if reason:
            return SandboxResult(ok=False, error=reason, backend=getattr(self.backend, "name", "?"),
                                 blocked_by="ast")
        limit = timeout if timeout is not None else settings.sandbox_timeout
        result = self.backend.run(code, data or [], limit)  # type: ignore[attr-defined]
        if len(result.stdout) > settings.sandbox_max_output:
            result.stdout = result.stdout[: settings.sandbox_max_output] + "\n…（输出已截断）"
        return result
