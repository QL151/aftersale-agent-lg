"""测试入口。

两种跑法，随便挑：
    python -m tests.run            # 零依赖（不需要装 pytest）
    pytest tests/ -q               # 装了 pytest 就用真 pytest

参数可以过滤文件名：
    python -m tests.run router graph
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.minipytest import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run(keywords=sys.argv[1:]))
