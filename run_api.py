"""启动 HTTP 服务。

    python run_api.py                     # 默认 http://127.0.0.1:8077
    python run_api.py --port 9000 --reload

打开浏览器访问根路径可以看到演示控制台：
左边提交工单并实时看 Agent 的执行过程，右边处理待人工审批的关键动作。
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="电商售后业务 Agent 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--reload", action="store_true", help="开发模式：改代码自动重启")
    args = parser.parse_args()

    import uvicorn

    print("=" * 78)
    print(f"电商售后业务 Agent 已启动：http://{args.host}:{args.port}")
    print(f"  · 演示控制台      http://{args.host}:{args.port}/")
    print(f"  · 能力体检        http://{args.host}:{args.port}/health")
    print(f"  · 待人工审批      http://{args.host}:{args.port}/approvals")
    print(f"  · 指标            http://{args.host}:{args.port}/metrics")
    print(f"  · Bad Case 扫描   http://{args.host}:{args.port}/bad-cases")
    print("=" * 78)
    uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
