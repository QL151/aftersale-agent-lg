"""一键启动菜单。

为什么菜单写在 Python 里，而不是直接写在 .bat 里：
    cmd 解析 .bat 用的是系统 OEM 代码页（中文 Windows 是 GBK / 936），
    而 .bat 文件本身是 UTF-8 保存的 —— 中文会被按 GBK 解码成乱码，
    连命令行都会被拆错、报「不是内部或外部命令」。
    所以 .bat 只保留纯 ASCII 的引导逻辑（切代码页 + 调本脚本），
    所有中文交互交给 Python —— Python 输出走 UTF-8，配合 chcp 65001 完全正常。
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable          # 用启动本脚本的那个解释器，和 .bat 里选的保持一致
WEB_URL = "http://127.0.0.1:8077"


def run(args) -> int:
    """在项目目录下跑一个脚本，等它结束。"""
    return subprocess.run([PY] + args, cwd=BASE).returncode


def open_browser_later(seconds: float = 3.0) -> None:
    """延迟开浏览器 —— 给 uvicorn 一点启动时间。"""
    def _job() -> None:
        time.sleep(seconds)
        webbrowser.open(WEB_URL)

    threading.Thread(target=_job, daemon=True).start()


def menu() -> str:
    print()
    print("=" * 64)
    print("   电商售后业务 Agent   （LangGraph + Redis 版）")
    print("=" * 64)
    print("    [1]  网页控制台       浏览器里提交工单 / 审批 / 看指标")
    print("    [2]  交互式命令行     输入工单即执行，适合连续试")
    print("    [3]  环境体检         看哪些能力是真跑的、哪些是降级")
    print("    [4]  查看可用订单     不知道该用哪个订单号时看这个")
    print("    [5]  重建演示数据     清空并重新生成 900 条订单")
    print("    [0]  退出")
    print("=" * 64)
    return input("请输入序号并回车 > ").strip()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    while True:
        try:
            choice = menu()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "1":
            print(f"\n正在启动网页控制台：{WEB_URL}")
            print("浏览器 3 秒后自动打开；关闭本窗口即停止服务。\n")
            open_browser_later()
            run(["run_api.py"])

        elif choice == "2":
            run(["demo_cli.py"])

        elif choice == "3":
            print()
            run(["run_cli.py", "--doctor"])
            input("\n回车返回菜单...")

        elif choice == "4":
            print()
            run(["data_map.py"])
            input("\n回车返回菜单...")

        elif choice == "5":
            answer = input(
                "\n这会清空并重建演示数据库（已有的工单、审批记录会丢失）。"
                "确认？(y/N) "
            ).strip().lower()
            if answer == "y":
                print()
                run(["run_cli.py", "--reset-db"])
            input("\n回车返回菜单...")

        elif choice in ("0", "q", "quit", "exit", ""):
            break

        else:
            print("没听清，请输入 0-5。")

    print("已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
