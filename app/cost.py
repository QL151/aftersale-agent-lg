"""成本熔断：单会话 token 上限 + 超限自动降级到小模型。

为什么 Agent 项目特别需要这个？
  普通接口的 token 消耗是可预测的（一问一答）；
  Agent 的消耗是**模型自己决定的** —— 它可能循环 8 轮、每轮把全部工具描述塞进上下文。
  没有上限，一个卡住的会话就能烧掉一天的预算。

三层熔断（递进，不是一刀切）：
  ① 达到 warn_ratio（默认 80%）      → 记 WARNING，开始用精简提示词
  ② 达到 100%                        → **降级到小模型**（便宜 5~10 倍），并记录降级标记
  ③ 降级后仍超 run 预算              → 停止本轮，转为人工工单（不硬撑）

这三层都写进 Trace，所以"成本异常"的会话在 Bad Case 报告里能一眼筛出来。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import settings

# 各模型的单价（元 / 1k token）。真实项目里从配置中心下发，这里给个合理默认。
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.001, 0.002),
    "deepseek-reasoner": (0.004, 0.016),
    "qwen-turbo": (0.0003, 0.0006),
    "qwen-plus": (0.0008, 0.002),
    "gpt-4o-mini": (0.0011, 0.0044),
    "gpt-4o": (0.018, 0.072),
    "mock": (0.0, 0.0),
}


def price_of(model: str) -> tuple[float, float]:
    if model in MODEL_PRICES:
        return MODEL_PRICES[model]
    lowered = model.lower()
    for name, prices in MODEL_PRICES.items():
        if name in lowered:
            return prices
    if "turbo" in lowered or "mini" in lowered or "small" in lowered:
        return (settings.price_small_per_1k, settings.price_small_per_1k * 2)
    return (settings.price_prompt_per_1k, settings.price_completion_per_1k)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = price_of(model)
    return (prompt_tokens / 1000.0) * prompt_price + (completion_tokens / 1000.0) * completion_price


@dataclass
class BudgetState:
    run_tokens: int = 0
    session_tokens: int = 0
    run_cost: float = 0.0
    session_cost: float = 0.0
    downgraded: bool = False
    warned: bool = False
    stopped: bool = False

    @property
    def run_ratio(self) -> float:
        limit = max(settings.run_token_budget, 1)
        return self.run_tokens / limit

    @property
    def session_ratio(self) -> float:
        limit = max(settings.session_token_budget, 1)
        return self.session_tokens / limit

    @property
    def ratio(self) -> float:
        """整体预算占用率（取更紧的那个）。"""
        return max(self.run_ratio, self.session_ratio)

    def to_dict(self) -> dict:
        return {
            "run_tokens": self.run_tokens,
            "session_tokens": self.session_tokens,
            "run_cost": round(self.run_cost, 6),
            "session_cost": round(self.session_cost, 6),
            "ratio": round(self.ratio, 4),
            "downgraded": self.downgraded,
            "warned": self.warned,
            "stopped": self.stopped,
        }


@dataclass
class ChargeOutcome:
    cost: float = 0.0
    action: str = "ok"                 # ok / warn / downgrade / stop
    reason: str = ""
    state: BudgetState = field(default_factory=BudgetState)


class CostController:
    """一次运行的预算控制器。会话级用量从会话记忆里读写，所以能跨轮累计。"""

    def __init__(self, session_id: str = "", memory_service=None, session_tokens: int = 0,
                 session_cost: float = 0.0) -> None:
        self.session_id = session_id
        self.memory = memory_service
        self.state = BudgetState(session_tokens=session_tokens, session_cost=session_cost)
        self.events: list[str] = []

    # ------------------------------------------------------------ 模型选择
    def pick_model(self, primary: str) -> str:
        """超预算就换成小模型。这是"成本熔断"最直接的落地。"""
        if self.state.downgraded and settings.cost_downgrade:
            small = settings.llm_small_model
            return small if small else primary
        return primary

    @property
    def prompt_budget_hint(self) -> int:
        """给上层一个"还能塞多少上下文"的提示：超 80% 就开始精简。"""
        remaining = settings.run_token_budget - self.state.run_tokens
        return max(remaining, 500)

    # ------------------------------------------------------------ 记账
    def charge(self, model: str, prompt_tokens: int, completion_tokens: int) -> ChargeOutcome:
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        self.state.run_tokens += prompt_tokens + completion_tokens
        self.state.session_tokens += prompt_tokens + completion_tokens
        self.state.run_cost += cost
        self.state.session_cost += cost

        action, reason = "ok", ""
        if self.state.run_ratio >= 1.0 or self.state.session_ratio >= 1.0:
            if settings.cost_downgrade and not self.state.downgraded:
                self.state.downgraded = True
                action = "downgrade"
                reason = (
                    f"预算已用 {self.state.ratio * 100:.0f}%，"
                    f"自动降级到小模型 {settings.llm_small_model}"
                )
            else:
                self.state.stopped = True
                action = "stop"
                reason = f"预算已用 {self.state.ratio * 100:.0f}%，停止本轮并转人工"
        elif self.state.ratio >= settings.cost_warn_ratio and not self.state.warned:
            self.state.warned = True
            action = "warn"
            reason = f"预算已用 {self.state.ratio * 100:.0f}%，进入精简模式"

        if reason:
            self.events.append(reason)
        self._persist()
        return ChargeOutcome(cost=cost, action=action, reason=reason, state=self.state)

    def _persist(self) -> None:
        """把会话级累计写回会话记忆（Redis / 本地），下一条消息才能接着算。"""
        if self.memory is None or not self.session_id:
            return
        try:
            memory = self.memory.load(self.session_id)
            memory.usage = {
                "tokens": self.state.session_tokens,
                "cost_yuan": round(self.state.session_cost, 6),
            }
            self.memory.store.save(memory)
        except Exception:  # noqa: BLE001 —— 记账失败不能影响业务
            pass

    # ------------------------------------------------------------ 展示
    def line(self) -> str:
        return (
            f"tokens={self.state.run_tokens}/{settings.run_token_budget}"
            f"（会话累计 {self.state.session_tokens}/{settings.session_token_budget}）"
            f" 成本=¥{self.state.run_cost:.4f}"
            f"{' [已降级]' if self.state.downgraded else ''}"
        )


def session_usage(memory_service, session_id: str) -> tuple[int, float]:
    if memory_service is None or not session_id:
        return 0, 0.0
    try:
        memory = memory_service.load(session_id)
        return int(memory.usage.get("tokens", 0)), float(memory.usage.get("cost_yuan", 0.0))
    except Exception:  # noqa: BLE001
        return 0, 0.0
