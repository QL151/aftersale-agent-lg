"""数据地图：一眼看清数据库里有什么、该用哪些订单号测试。

用法：
    python data_map.py              # 打印全部场景与对应订单号
    python data_map.py --ticket     # 只看工单/客户
    python data_map.py --recheck    # 只列当前仍然可退的订单（推荐测试用）

为什么需要这个工具？
    数据是 seed.py 随机生成的，订单号无法预先知道。
    评测脚本 eval/run_eval.py 里有一套"订单池"机制按条件自动挑单，
    这个脚本把同一套逻辑做成人类可读的清单。
"""

from __future__ import annotations

import argparse
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from app.db import Repo, connect, init_schema, is_seeded, check_policy
from app.config import settings
from app.seed import seed


def ensure_data() -> Repo:
    conn = connect(settings.db_path)
    init_schema(conn)
    if not is_seeded(conn):
        conn.close()
        seed()
    else:
        conn.close()
    return Repo()


def collect(repo: Repo, predicate, limit: int = 8) -> list:
    rows = repo.query("SELECT order_id FROM orders ORDER BY order_id")
    out = []
    for r in rows:
        o = repo.get_order(r["order_id"])
        if o and predicate(o):
            out.append(o)
            if len(out) >= limit:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="数据地图")
    ap.add_argument("--ticket", action="store_true", help="只看工单/客户")
    ap.add_argument("--recheck", action="store_true", help="只列当前仍可退的订单")
    args = ap.parse_args()

    repo = ensure_data()
    orders = [repo.get_order(r["order_id"])
              for r in repo.query("SELECT order_id FROM orders ORDER BY order_id")]
    orders = [o for o in orders if o]

    def allowed(o):
        v = check_policy(o, repo.get_rule(o.category))
        return v.allowed and not v.needs_manual

    def show(title: str, items, note: str = ""):
        print(f"\n{'=' * 74}")
        print(f"{title}  （共 {len(items)} 个）")
        if note:
            print(f"  → {note}")
        print("=" * 74)
        if not items:
            print("  （无）")
            return
        for o in items:
            opened = "已拆封" if o.is_opened else "未拆封"
            extra = f" 实付 {o.amount} 元"
            print(f"  {o.order_id}  |  {o.category:<4} | {o.sku_name:<16} | "
                  f"{o.status:<5} | {opened}{extra}")

    if args.recheck or not args.ticket:
        print("\n" + "█" * 74)
        print("  订单号速查表（数据由 app/seed.py 随机生成，此表实时查询得出）")
        print("█" * 74)

    if args.recheck:
        show("★ 可退 + 金额 <= 200（小额免打扰，自动放行）",
             collect(repo, lambda o: o.status == "已签收" and o.amount <= 200
                     and o.is_opened == 0 and allowed(o)),
             "申请退款会直接通过，不触发审批")
        show("★ 可退 + 金额 > 300（触发人工审批）",
             collect(repo, lambda o: o.status == "已签收" and o.amount > 300
                     and o.is_opened == 0 and allowed(o)),
             "申请退款会中断等人批准 ← 演示审批用这些")
        show("★ 已签收但超期 / 不符合政策（会被 deny 拒绝）",
             collect(repo, lambda o: o.status == "已签收" and o.delivered_at
                     and not check_policy(o, repo.get_rule(o.category)).allowed),
             "申请退款会被政策校验直接拒绝")
        return 0

    if not args.ticket:
        show("[状态] 已支付（未发货）", collect(repo, lambda o: o.status == "已支付"))
        show("[状态] 已发货（在途）", collect(repo, lambda o: o.status == "已发货"))
        show("[状态] 已签收", collect(repo, lambda o: o.status == "已签收"))
        show("[状态] 已取消", collect(repo, lambda o: o.status == "已取消"))
        show("[状态] 已退款（重复申请会被拒）", collect(repo, lambda o: o.status == "已退款"))

        show("[场景] 可退 + 金额 <= 200（自动放行）",
             collect(repo, lambda o: o.status == "已签收" and o.amount <= 200
                     and o.is_opened == 0 and allowed(o)))
        show("[场景] 可退 + 金额 > 300（★ 触发人工审批）",
             collect(repo, lambda o: o.status == "已签收" and o.amount > 300
                     and o.is_opened == 0 and allowed(o)))
        show("[场景] 数码已拆封（不支持无理由）",
             collect(repo, lambda o: o.status == "已签收" and o.category == "数码"
                     and o.is_opened == 1))
        show("[场景] 服饰未拆封（可无理由退货）",
             collect(repo, lambda o: o.status == "已签收" and o.category == "服饰"
                     and o.is_opened == 0 and allowed(o)))

        abn = repo.query("SELECT order_id, carrier, abnormal_reason FROM shipments "
                         "WHERE status='异常' LIMIT 8")
        print(f"\n{'=' * 74}")
        print(f"[场景] 物流异常件（共 {len(abn)} 个）")
        print("=" * 74)
        for r in abn:
            print(f"  {r['order_id']}  |  {r['carrier']} | {r.get('abnormal_reason','')}")

    print(f"\n{'=' * 74}")
    print("[基础] 客户 / 商品 / 规则 / 知识库")
    print("=" * 74)
    custs = repo.query("SELECT customer_id, name, level FROM customers LIMIT 5")
    for c in custs:
        print(f"  客户 {c['customer_id']}  |  {c['name']:<6} | 等级 {c.get('level','')}")
    rules = repo.all_rules()
    print(f"\n  售后规则 {len(rules)} 条：")
    for r in rules:
        print(f"    {r['category']:<4} | {r['clause']:<12} | "
              f"无理由 {r['max_days_after_delivery']} 天 | "
              f"允许拆封 {'是' if r['allow_opened'] else '否'}")
    kb = repo.query("SELECT id, title FROM kb_articles")
    print(f"\n  知识库 {len(kb)} 篇：")
    for a in kb:
        print(f"    {a['id']} | {a['title']}")

    print(f"\n{'=' * 74}")
    print("  提示：退款成功后订单会变成「已退款」，重复测试请换新订单号")
    print("  重新生成全套数据：python -c \"from app.seed import seed; seed(force=True)\"")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
