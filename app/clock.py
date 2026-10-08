"""可注入的时钟。

为什么单独抽出来？
  售后业务的判定几乎全是"时间差"：签收后几天、价保几天内、退款时效。
  如果代码里直接调 datetime.now()，测试就只能靠"跑的时候刚好是第几天"——非常脆。
  把"业务当前时间"做成可覆盖（BIZ_NOW 环境变量），
  规则测试和评测集就变成**完全确定性**的，这也让 Bad Case 可以被精确复现。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

_FMT = "%Y-%m-%d %H:%M:%S"


def now() -> datetime:
    """业务当前时间。BIZ_NOW 有值就用它（测试/评测用），否则用真实时间。"""
    pinned = (os.getenv("BIZ_NOW") or "").strip()
    if pinned:
        for fmt in (_FMT, "%Y-%m-%d"):
            try:
                return datetime.strptime(pinned, fmt)
            except ValueError:
                continue
    return datetime.now()


def today_str() -> str:
    return now().strftime("%Y-%m-%d")


def stamp() -> str:
    return now().strftime(_FMT)


def parse(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in (_FMT, "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def days_since(value: str) -> float | None:
    """从某时间点到"业务当前时间"过了多少天。None = 无法解析。"""
    start = parse(value)
    if start is None:
        return None
    return (now() - start).total_seconds() / 86400.0


def ago(days: float) -> str:
    return (now() - timedelta(days=days)).strftime(_FMT)


def ahead(days: float) -> str:
    return (now() + timedelta(days=days)).strftime(_FMT)
