"""大模型客户端：真实模型（OpenAI 兼容）+ 离线假模型（MockLLM）。

为什么必须有一个假模型？
  Agent 的行为测试如果每次真调模型，就同时踩三个坑：慢、贵、**不可复现**（输出有随机性）。
  有了离线假模型：
    · 别人 clone 下来没有 Key 也能跑通全流程、看到完整 Trace
    · 评测集可以进 CI，断言稳定
  这是 Agent 项目与普通项目的关键工程差异。

  注意：因为高频意图走的是**确定性工作流（不经过模型）**，
  所以假模型只需要覆盖"长尾自主 Agent"和"路由兜底分类"两条路径，实现成本很低 ——
  这也是"路由层省钱"的另一个好处：连测试都变便宜了。
"""

from __future__ import annotations

import json
import re
import time

from .config import Settings, settings as global_settings
from .schemas import LLMReply, ToolCall


class LLMClient:
    """真实模型客户端（DeepSeek / Qwen / 任意 OpenAI 兼容网关）。"""

    name = "openai-compatible"

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.s.llm_api_key, base_url=self.s.llm_base_url, timeout=self.s.llm_timeout
            )
        return self._client

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMReply:
        client = self._get_client()
        target_model = model or self.s.llm_model
        started = time.time()
        response = client.chat.completions.create(
            model=target_model,
            messages=messages,
            tools=tools or None,
            temperature=self.s.llm_temperature if temperature is None else temperature,
        )
        duration_ms = int((time.time() - started) * 1000)
        message = response.choices[0].message

        calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            raw = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {"__raw__": raw}      # 参数不是合法 JSON → 交给 Guardrail 拦
            calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments, raw_arguments=raw)
            )

        usage = getattr(response, "usage", None)
        return LLMReply(
            text=message.content or "",
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            duration_ms=duration_ms,
            model=target_model,
        )


class MockLLM:
    """离线假模型：用规则模拟"看任务 → 选工具 → 看结果 → 收尾"。

    它不聪明，但**足够把长尾自主 Agent 的流程走完整**，
    并且输出完全确定 —— 评测指标因此可复现。
    """

    name = "mock"

    KNOWN_CUSTOMERS = ["张三", "李四", "王五", "赵伟", "钱芳", "孙娜"]

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.calls = 0

    # ------------------------------------------------------------ 接口
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMReply:
        started = time.time()
        self.calls += 1
        target_model = model or self.s.llm_model
        available = {t["function"]["name"] for t in (tools or [])}
        task = " ".join(m["content"] for m in messages if m.get("role") == "user")
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        done = [m.get("name", "") for m in tool_messages]
        last = tool_messages[-1]["content"] if tool_messages else ""

        def reply(text: str = "", calls: list[ToolCall] | None = None) -> LLMReply:
            return LLMReply(
                text=text,
                tool_calls=calls or [],
                duration_ms=max(int((time.time() - started) * 1000), 1),
                model=target_model,
                prompt_tokens=self._approx_tokens(messages),
                completion_tokens=len(text) // 3 + 12 * len(calls or []),
            )

        def call(name: str, **arguments) -> LLMReply:
            if name not in available:
                return reply(f"[离线模拟] 可用工具里没有 {name}，我无法完成这个操作。")
            return reply(calls=[ToolCall(id=f"mock-{self.calls}-{len(done) + 1}",
                                         name=name, arguments=arguments)])

        def order_id() -> str:
            found = re.search(r"(SO\d{6,})", task)
            return found.group(1) if found else ""

        def customer_id() -> str:
            found = re.search(r"(C\d{5})", task)
            return found.group(1) if found else ""

        def customer_name() -> str:
            for name in self.KNOWN_CUSTOMERS:
                if name in task:
                    return name
            return ""

        # ---- 被护栏拦下：换安全做法，或如实说明 ----
        if last.startswith("[已拦截]") or last.startswith("[沙箱拒绝"):
            if "run_analysis_code" in done and "search_knowledge_base" not in done:
                return call("search_knowledge_base", query=task[:60])
            return reply(
                "这个操作被安全策略拦下了，我不会绕过它。\n"
                f"拦截原因：{last.strip()[:200]}\n"
                "如果你确实需要，请人工确认后再操作。"
            )

        # ---- 越权/破坏性请求：先试只读 SQL，让护栏有机会拦下并演示"可恢复的护栏" ----
        if any(k in task for k in ("删掉", "删除", "清空", "drop", "DROP", "truncate", "改一下表")):
            if "query_business_db" in available and "query_business_db" not in done:
                return call("query_business_db", sql="DROP TABLE orders")
            return reply(
                "这个请求涉及删除/修改数据，超出了我的权限范围，我不会绕过安全策略执行。\n"
                "如果确实需要清理数据，请由 DBA 走变更流程。"
            )

        # ---- 长尾场景：按关键词排好顺序，只调用"参数齐全"的工具 ----
        plan: list[tuple[str, dict]] = []
        wants_analysis = any(
            k in task for k in ("分析", "比例", "占比", "统计", "分布", "趋势", "排名", "异常率")
        )
        if wants_analysis:
            plan.append(("run_analysis_code", {
                "code": "from collections import Counter\n"
                        "rows = DATA\n"
                        "key = 'intent' if rows and 'intent' in rows[0] else 'category'\n"
                        "c = Counter(r.get(key) for r in rows)\n"
                        "total = len(rows) or 1\n"
                        "print('分布：')\n"
                        "for k, v in c.most_common():\n"
                        "    print(f'  {k}: {v} ({v/total*100:.1f}%)')\n"
                        "RESULT = dict(c)",
                "dataset": "tickets",
            }))

        if order_id():
            plan.append(("get_order_detail", {"order_id": order_id()}))
        elif customer_name():
            plan.append(("search_orders_by_customer", {"customer_name": customer_name()}))

        # 注意：这些工具都有必填参数，参数拿不到就不要调 ——
        # 否则会被 Guardrail 按参数校验拦下，模型却以为自己"试过了"。
        if order_id() and any(k in task for k in ("物流", "到哪", "快递", "运单", "没收到")):
            plan.append(("get_logistics_trace", {"order_id": order_id()}))
        if any(k in task for k in ("退货", "换货", "寄回")):
            plan.append(("search_knowledge_base", {"query": "退货换货 操作指引"}))
        if customer_id() and any(k in task for k in ("画像", "等级", "历史", "偏好")):
            plan.append(("get_customer_profile", {"customer_id": customer_id()}))
        if any(k in task for k in ("规则", "制度", "政策", "条款")) and not any(
            n == "search_knowledge_base" for n, _ in plan
        ):
            plan.append(("search_knowledge_base", {"query": task[:60]}))
        if order_id() and any(k in task for k in ("建单", "工单", "登记", "跟进")):
            plan.append(("create_ticket", {
                "order_id": order_id(), "intent": "长尾咨询", "title": task[:60],
            }))

        for name, arguments in plan:
            if name not in done:
                return call(name, **arguments)

        # ---- 收尾 ----
        if tool_messages:
            blocked = sum(
                1 for m in tool_messages
                if m["content"].startswith(("[已拦截]", "[沙箱拒绝"))
            )
            note = f"\n（本次有 {blocked} 次操作被安全策略拦截，详见 Trace）" if blocked else ""
            return reply(
                "[离线模拟回答] 已按步骤处理完成。\n"
                f"调用过程：{'、'.join(done)}。\n"
                f"最后一步结果：\n{last[:500]}{note}"
            )
        return reply(
            "[离线模拟回答] 这个问题不需要调用工具，我直接回答："
            "请补充订单号或说明具体诉求，我可以帮你查订单、物流、退款规则或建工单。"
        )

    @staticmethod
    def _approx_tokens(messages: list[dict]) -> int:
        total = 0
        for message in messages:
            content = message.get("content") or ""
            total += len(str(content)) // 2 + 8
        return total


def build_llm(settings: Settings | None = None):
    """按配置选择真实模型或离线假模型。"""
    active = settings or global_settings
    if active.use_mock:
        return MockLLM(active)
    return LLMClient(active)
