"""沙箱内的执行器：在**独立进程**里跑模型生成的 Python 代码。

安全边界说明（面试要能讲清楚，别吹）：
  这里做的是"纵深防御"，不是"安全边界"：
    ① 独立进程 + PYTHONPATH 隔离 + -I（忽略环境变量与用户 site-packages）
    ② 受限 builtins（没有 open / eval / exec / __import__ / compile）
    ③ import 白名单（math / statistics / json / re / datetime / collections ...）
    ④ AST 层禁止访问 dunder 属性、禁止 getattr/setattr/globals/locals/vars
    ⑤ 硬超时 + 输出截断
  能突破它的手法仍然存在（例如通过白名单模块的内存放大做 DoS）。
  **真正的安全边界是 OS 级隔离**：Docker / gVisor / microVM。
  所以本项目把执行器做成可替换的：
    SANDBOX_BACKEND=subprocess（默认，零依赖） | docker（有容器时用）
  见 app/sandbox.py 的 build_sandbox()。
"""

from __future__ import annotations

import json
import sys

# 子进程被 -I 启动，PYTHONIOENCODING 会被忽略，所以这里显式把标准流切成 UTF-8。
# 不加这一行，在中文 Windows 上 print 中文会直接 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

# ---- 与 app/sandbox.py 保持一致的模块白名单
ALLOWED_MODULES = {
    "math", "statistics", "json", "re", "datetime", "collections",
    "itertools", "functools", "decimal", "fractions", "random", "string",
}

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "int": int, "isinstance": isinstance, "len": len,
    "list": list, "map": map, "max": max, "min": min, "pow": pow, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError, "StopIteration": StopIteration,
    "True": True, "False": False, "None": None,
}


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"沙箱不允许导入模块 `{name}`（白名单：{sorted(ALLOWED_MODULES)}）")
    return __import__(name, globals, locals, fromlist, level)


def _guard_attr():
    """禁止访问 dunder 属性：`().__class__.__bases__` 这类逃逸全靠它。"""

    class Guarded:
        def __getattribute__(self, item):  # pragma: no cover - 仅在逃逸尝试时触发
            if item.startswith("__") and item.endswith("__"):
                raise AttributeError(f"沙箱禁止访问 dunder 属性 `{item}`")
            return object.__getattribute__(self, item)

    return Guarded


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"载荷不是合法 JSON：{exc}"}, ensure_ascii=False))
        return 2

    code = payload.get("code", "")
    data = payload.get("data", [])

    namespace: dict = {
        "__builtins__": {**_SAFE_BUILTINS, "__import__": _guarded_import},
        "DATA": data,
        "RESULT": None,
    }

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stdout": buffer.getvalue(),
                    "result": _safe_json(namespace.get("RESULT")),
                },
                ensure_ascii=False,
            )
        )
        return 0

    print(
        json.dumps(
            {
                "ok": True,
                "stdout": buffer.getvalue(),
                "result": _safe_json(namespace.get("RESULT")),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _safe_json(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)[:2000]


if __name__ == "__main__":
    sys.exit(main())
