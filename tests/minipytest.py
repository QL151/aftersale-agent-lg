"""零依赖测试运行器：没有 pytest 也能跑完整套件。

为什么自己写一个？
  本项目的核心约束是"任何人 clone 下来就能跑起来"。
  如果测试必须 `pip install pytest` 才能跑，那在没有外网的环境里（CI 沙箱、内网机器、
  面试时对方的电脑）就**无法验证任何东西** —— 这恰恰是招聘方最想看的。
  所以这里实现了 pytest 够用的那一小部分：fixture（含 yield 与 scope）、
  parametrize、approx、raises、importorskip、monkeypatch、tmp_path。

  装了真 pytest 时，本模块**不会**遮蔽它（见 install_shim 的判定），
  你依然可以用 `pytest tests/` 跑同一批用例。

用法：
    python -m tests.run              # 跑全部
    python -m tests.run router graph # 只跑名字里含这些关键字的文件
"""

from __future__ import annotations

import inspect
import os
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent


# ==================================================================== pytest 垫片


class Skipped(Exception):
    pass


class _Approx:
    def __init__(self, expected: Any, rel: float = 1e-6, abs: float = 1e-12) -> None:  # noqa: A002
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _close(self, a, b) -> bool:
        try:
            return abs(a - b) <= max(self.abs, self.rel * max(abs(a), abs(b)))
        except TypeError:
            return a == b

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (list, tuple)) and isinstance(self.expected, (list, tuple)):
            return len(other) == len(self.expected) and all(
                self._close(o, e) for o, e in zip(other, self.expected)
            )
        return self._close(other, self.expected)

    def __repr__(self) -> str:
        return f"approx({self.expected!r})"


class _Raises:
    def __init__(self, expected: Any) -> None:
        self.expected = expected
        self.value: BaseException | None = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected}")
        if isinstance(self.expected, tuple):
            matched = issubclass(exc_type, self.expected)
        else:
            matched = issubclass(exc_type, self.expected)
        if not matched:
            return False
        self.value = exc
        return True


class _Mark:
    def parametrize(self, argnames: Any, argvalues: Any, **kwargs: Any):
        names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def decorator(func: Callable) -> Callable:
            existing = list(getattr(func, "_parametrize", []))
            existing.append((names, list(argvalues)))
            func._parametrize = existing
            return func

        return decorator

    def __getattr__(self, _name: str):
        def decorator(*args: Any, **kwargs: Any):
            if args and callable(args[0]):
                return args[0]

            def inner(func: Callable) -> Callable:
                return func

            return inner

        return decorator


def _fixture(*args: Any, **kwargs: Any):
    """支持 @fixture、@fixture()、@fixture(scope=..., autouse=...) 三种写法。"""
    def decorate(func: Callable) -> Callable:
        func._fixture = {
            "scope": kwargs.get("scope", "function"),
            "autouse": bool(kwargs.get("autouse", False)),
            "name": kwargs.get("name") or func.__name__,
        }
        return func

    if args and callable(args[0]) and not kwargs:
        return decorate(args[0])
    return decorate


def _importorskip(name: str, *_args: Any, **_kwargs: Any):
    try:
        return __import__(name)
    except ImportError as exc:
        raise Skipped(f"缺少模块 {name}") from exc


def _skip(reason: str = "") -> None:
    raise Skipped(reason)


def _fail(reason: str = "") -> None:
    raise AssertionError(reason)


def install_shim() -> bool:
    """只有在真 pytest 不存在时才安装垫片。返回是否安装了垫片。"""
    try:
        import pytest  # noqa: F401
        return False
    except ImportError:
        pass

    shim = types.ModuleType("pytest")
    shim.fixture = _fixture
    shim.mark = _Mark()
    shim.approx = lambda expected, rel=1e-6, abs=1e-12: _Approx(expected, rel, abs)  # noqa: A002
    shim.raises = lambda expected, **kwargs: _Raises(expected)
    shim.importorskip = _importorskip
    shim.skip = _skip
    shim.fail = _fail
    shim.Skipped = Skipped
    shim.__version__ = "mini-0.1"
    sys.modules["pytest"] = shim
    return True


# ==================================================================== 夹具


class MonkeyPatch:
    def __init__(self) -> None:
        self._env: list[tuple[str, str | None]] = []
        self._attrs: list[tuple[Any, str, Any]] = []

    def setenv(self, name: str, value: str) -> None:
        self._env.append((name, os.environ.get(name)))
        os.environ[name] = str(value)

    def delenv(self, name: str, raising: bool = True) -> None:
        if name not in os.environ and raising:
            raise KeyError(name)
        self._env.append((name, os.environ.get(name)))
        os.environ.pop(name, None)

    def setattr(self, target: Any, name: str, value: Any) -> None:
        self._attrs.append((target, name, getattr(target, name, None)))
        setattr(target, name, value)

    def undo(self) -> None:
        for name, old in reversed(self._env):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        for target, name, old in reversed(self._attrs):
            setattr(target, name, old)
        self._env.clear()
        self._attrs.clear()


def fixture_meta(obj: Any) -> dict | None:
    """读取夹具元信息，三种装饰器都认。

    为什么不能只看 `_fixture`：install_shim() 在装了真 pytest 时会主动让位
    （不遮蔽它），此时 conftest.py 里 `@pytest.fixture` 打的是 pytest 自己的标记，
    本模块的 `_fixture` 根本不存在 ——
    结果就是所有依赖夹具的用例集体报"没有名为 xxx 的夹具"。

    标记位置随 pytest 版本变过：
      · pytest <= 8：函数属性 `_pytestfixturefunction`
      · pytest >= 9：包装对象上的 `_fixture_function_marker`（装饰器不再返回原函数）
    """
    info = getattr(obj, "_fixture", None)
    if info:
        return dict(info)
    marker = (getattr(obj, "_fixture_function_marker", None)
              or getattr(obj, "_pytestfixturefunction", None))
    if marker is not None:
        return {
            "scope": getattr(marker, "scope", None) or "function",
            "autouse": bool(getattr(marker, "autouse", False)),
            "name": getattr(marker, "name", None) or getattr(obj, "__name__", ""),
        }
    return None


def unwrap_fixture(obj: Any) -> Any:
    """取回真正干活的原函数。

    pytest >= 9 的 @fixture 返回的是 FixtureFunctionDefinition 包装对象，
    直接调用它会被 pytest 拦下并报 "Fixture called directly"，
    原函数挂在 `_fixture_function` 上。
    """
    return getattr(obj, "_fixture_function", obj)


class FixtureManager:
    """按名字解析夹具，支持 yield 拆解与 scope 缓存。"""

    def __init__(self, module: types.ModuleType) -> None:
        self.registry: dict[str, Callable] = {}
        for name, obj in vars(module).items():
            info = fixture_meta(obj)
            if info:
                self.registry[info["name"]] = obj
        self.cache: dict[tuple[str, str], Any] = {}
        self.session_teardowns: list[Callable] = []

    def wants(self, func: Callable) -> list[str]:
        return [
            name for name, param in inspect.signature(func).parameters.items()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        ]

    def autouse_names(self, scope: str) -> list[str]:
        out = []
        for name, func in self.registry.items():
            info = fixture_meta(func) or {}
            if info.get("autouse") and info.get("scope") == scope:
                out.append(name)
        return out

    def resolve(self, name: str, scope_key: str, teardowns: list[Callable]) -> Any:
        if name == "tmp_path":
            path = Path(tempfile.mkdtemp(prefix="minipytest-"))
            teardowns.append(lambda: shutil.rmtree(path, ignore_errors=True))
            return path
        if name == "monkeypatch":
            patch = MonkeyPatch()
            teardowns.append(patch.undo)
            return patch
        if name == "request":
            return None

        func = self.registry.get(name)
        if func is None:
            raise LookupError(f"没有名为 {name} 的夹具")

        info = fixture_meta(func)
        cache_key = (name, info["scope"] if info["scope"] != "function" else scope_key)
        if info["scope"] != "function" and cache_key in self.cache:
            return self.cache[cache_key]

        kwargs = {
            dep: self.resolve(dep, scope_key, teardowns)
            for dep in self.wants(func)
        }
        result = unwrap_fixture(func)(**kwargs)
        if inspect.isgenerator(result):
            generator = result
            value = next(generator)

            def finalize(gen=generator) -> None:
                try:
                    next(gen)
                except StopIteration:
                    pass

            if info["scope"] == "session":
                self.session_teardowns.append(finalize)
            else:
                teardowns.append(finalize)
            result = value

        if info["scope"] != "function":
            self.cache[cache_key] = result
        return result


# ==================================================================== 收集与执行


def parametrize_specs(func: Callable) -> list[tuple[list[str], list]]:
    """收集参数化声明，两种标记都认（理由同 fixture_meta）。"""
    specs: list[tuple[list[str], list]] = list(getattr(func, "_parametrize", []))

    marks = getattr(func, "pytestmark", None) or []
    if not isinstance(marks, (list, tuple)):
        marks = [marks]
    for mark in marks:
        if getattr(mark, "name", None) != "parametrize":
            continue
        args = list(getattr(mark, "args", ()) or ())
        if len(args) < 2:
            continue
        argnames, argvalues = args[0], args[1]
        names = ([n.strip() for n in argnames.split(",")]
                 if isinstance(argnames, str) else list(argnames))
        specs.append((names, list(argvalues)))
    return specs


def _expand(func: Callable) -> list[tuple[str, dict]]:
    """把 parametrize 展开成多个具体用例。"""
    params = parametrize_specs(func)
    cases: list[tuple[str, dict]] = [("", {})]
    for names, values in params:
        expanded: list[tuple[str, dict]] = []
        for suffix, kwargs in cases:
            for value in values:
                items = value if isinstance(value, (tuple, list)) and len(names) > 1 else (value,)
                mapping = dict(zip(names, items))
                label = "-".join(str(v)[:12] for v in items)
                expanded.append((f"{suffix}[{label}]" if suffix else f"[{label}]",
                                 {**kwargs, **mapping}))
        cases = expanded
    return cases


def run(tests_dir: Path = TESTS_DIR, keywords: list[str] | None = None) -> int:
    shim = install_shim()
    print("=" * 96)
    print(f"零依赖测试运行器（pytest 垫片：{'已启用' if shim else '未启用（检测到真 pytest）'}）")
    print("=" * 96)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(tests_dir))

    conftest_path = tests_dir / "conftest.py"
    if not conftest_path.exists():
        print(f"缺少 {conftest_path}")
        return 2

    import importlib.util

    spec = importlib.util.spec_from_file_location("conftest", conftest_path)
    conftest = importlib.util.module_from_spec(spec)
    sys.modules["conftest"] = conftest
    spec.loader.exec_module(conftest)

    fixtures = FixtureManager(conftest)
    # session 级 autouse 夹具（造数）立刻生效
    session_teardowns: list[Callable] = []
    for name in fixtures.autouse_names("session"):
        fixtures.resolve(name, "session", session_teardowns)

    passed = failed = skipped = 0
    failures: list[str] = []

    files = sorted(p for p in tests_dir.glob("test_*.py"))
    if keywords:
        files = [p for p in files if any(k in p.name for k in keywords)]
    if not files:
        print("没有匹配的测试文件。")
        return 1

    for path in files:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append(f"{path.name} 导入失败：{type(exc).__name__}: {exc}")
            print(f"\n[导入失败] {path.name}: {exc}")
            traceback.print_exc()
            continue

        module_fixtures = FixtureManager(module)
        # 模块自己的夹具优先，其次 conftest
        merged = FixtureManager(conftest)
        merged.registry.update(module_fixtures.registry)
        try:
            merged.registry.update(
                {n: f for n, f in fixtures.registry.items() if n not in merged.registry}
            )
        except AttributeError:
            pass

        print(f"\n--- {path.name} " + "-" * max(96 - len(path.name) - 4, 4))

        if not keywords:
            for name in merged.autouse_names("module"):
                merged.resolve(name, f"{path.stem}-module", [])

        targets: list[tuple[str, Callable]] = []
        for name, obj in sorted(vars(module).items()):
            if name.startswith("test_") and inspect.isfunction(obj):
                targets.append((name, obj))
            elif name.startswith("Test") and inspect.isclass(obj):
                instance = obj()
                for attr, member in sorted(vars(obj).items()):
                    if attr.startswith("test_") and inspect.isfunction(member):
                        targets.append((f"{name}::{attr}", getattr(instance, attr)))

        for label, func in targets:
            for suffix, kwargs in _expand(func):
                teardowns: list[Callable] = []
                title = f"{label}{suffix}"
                try:
                    call_kwargs = dict(kwargs)
                    for dep in merged.wants(func):
                        if dep not in call_kwargs:
                            call_kwargs[dep] = merged.resolve(
                                dep, f"{path.stem}-{title}", teardowns
                            )
                    for auto in merged.autouse_names("function"):
                        if auto not in call_kwargs:
                            merged.resolve(auto, f"{path.stem}-{title}-{auto}", teardowns)
                    func(**call_kwargs)
                    passed += 1
                    print(f"  PASS  {title}")
                except Skipped as exc:
                    skipped += 1
                    print(f"  SKIP  {title}  ({exc})")
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    detail = traceback.format_exc()
                    failures.append(f"{title}\n{detail}")
                    print(f"  FAIL  {title}: {type(exc).__name__}: {exc}")
                finally:
                    for finalize in reversed(teardowns):
                        try:
                            finalize()
                        except Exception:  # noqa: BLE001
                            pass

    for finalize in reversed(fixtures.session_teardowns):
        try:
            finalize()
        except Exception:  # noqa: BLE001
            pass

    print()
    print("=" * 96)
    print(f"结果：通过 {passed}｜失败 {failed}｜跳过 {skipped}")
    if failures:
        print("-" * 96)
        print("失败详情：")
        for item in failures:
            print(item)
    print("=" * 96)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run(keywords=sys.argv[1:]))
