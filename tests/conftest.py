"""pytest 公共配置。

关键点：**在 import 任何 app 模块之前**把路径指向临时目录。
因为 app.config.settings 是模块级单例，import 时就把配置读死了 ——
如果先 import 再改环境变量，测试就会写进真实业务库，把演示数据搞脏。
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

# 测试工作区固定放在项目内的 data/.test-tmp/ 下，而不是交给 tempfile 自己挑目录：
# 有些环境（CI 沙箱、某些 Windows 终端）TEMP/TMP 没设或指向不可写的位置，
# tempfile 会退化成"在当前目录建临时文件夹"，把项目根目录搞得到处是 aftersale-test-*。
_TEST_ROOT = Path(__file__).resolve().parent.parent / "data" / ".test-tmp"
_TEST_ROOT.mkdir(parents=True, exist_ok=True)
_TMP = Path(tempfile.mkdtemp(prefix="run-", dir=_TEST_ROOT))

# 会话结束自动清理，不给下次运行留垃圾
atexit.register(lambda: shutil.rmtree(_TMP, ignore_errors=True))

os.environ["MOCK_LLM"] = "true"                 # 测试一律用离线假模型：快、稳、可复现
os.environ["DB_PATH"] = str(_TMP / "test.db")
os.environ["TRACE_DIR"] = str(_TMP / "traces")
os.environ["STATE_DIR"] = str(_TMP / "state")
os.environ["MCP_SERVERS"] = ""                  # 默认不拉 MCP 子进程
os.environ["REDIS_ENABLED"] = "false"           # 强制走降级实现，保证测试环境无关
os.environ["POSTGRES_ENABLED"] = "false"
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["APPROVAL_AUTO_APPROVE_MAX"] = "200"
os.environ.pop("BIZ_NOW", None)

import pytest  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Repo, reset_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _seeded():
    """整个测试会话造一次数（900 订单 / 900 工单），避免每个用例都重建。"""
    from app.seed import seed

    settings.ensure_dirs()
    reset_db()
    seed(force=True)
    yield


@pytest.fixture()
def repo():
    with Repo() as item:
        yield item


@pytest.fixture()
def clean_memory():
    """每个用例用独立会话 ID，避免用例之间通过会话记忆互相影响。"""
    import uuid

    from app.memory.session import session_service

    service = session_service()
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    yield session_id
    service.clear(session_id)


@pytest.fixture()
def agent():
    from app.runner import AftersaleAgent

    return AftersaleAgent()


def pick_order(repo, predicate, limit=1):
    """按条件挑订单，让测试不依赖写死的订单号。"""
    found = []
    for row in repo.query("SELECT order_id FROM orders ORDER BY order_id"):
        order = repo.get_order(row["order_id"])
        if order and predicate(order):
            found.append(order)
            if len(found) >= limit:
                break
    return found


def refundable_order(repo, max_amount=200.0):
    from app.db import check_policy

    def ok(order):
        verdict = check_policy(order, repo.get_rule(order.category))
        return (order.status == "已签收" and order.amount <= max_amount
                and order.is_opened == 0 and verdict.allowed)

    return pick_order(repo, ok, 1)


def big_refundable_order(repo, min_amount=300.0):
    from app.db import check_policy

    def ok(order):
        verdict = check_policy(order, repo.get_rule(order.category))
        return (order.status == "已签收" and order.amount >= min_amount
                and order.is_opened == 0 and verdict.allowed)

    return pick_order(repo, ok, 1)
