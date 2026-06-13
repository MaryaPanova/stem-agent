"""Trading environment — the fully-implemented domain.

Note what is *not* here: nothing tells the agent it is a trader. Tasks state an objective
("maximise portfolio_value"); the agent has to discover, by calling tools, that there is a
price that moves, that it can buy and sell, and that time advances. Specialization into "a
trader" happens in the genome, through evolution — never in this file.
"""

from __future__ import annotations

from ...environment import BaseEnvironment, StepResult
from ...models import Task, TaskResult, ToolSpec, Trajectory
from .exchange import Exchange, make_prices, oracle_return

START_CASH = 1000.0

# (seed, n_visible) per task. Train and test use different seeds; the *pattern* (mean
# reversion) is shared, so a skill learned on train generalises to test.
_TRAIN = [(11, 24), (29, 28), (47, 24)]
_TEST = [(101, 26), (202, 24)]


def _tasks() -> list[Task]:
    tasks: list[Task] = []
    for seed, n in _TRAIN:
        tasks.append(_make_task(seed, n, "train"))
    for seed, n in _TEST:
        tasks.append(_make_task(seed, n, "test"))
    return tasks


def _make_task(seed: int, n_visible: int, split: str) -> Task:
    return Task(
        id=f"trade-{split}-{seed}",
        objective=(
            "Grow `portfolio_value` as much as you can before the final step. You begin "
            f"with {START_CASH:.0f} cash and no position."
        ),
        max_steps=max(40, (n_visible + 6) * 2),
        split=split,
        metadata={"seed": seed, "n_visible": n_visible},
    )


class TradingEnvironment(BaseEnvironment):
    name = "trading"

    def __init__(self) -> None:
        self._ex: Exchange | None = None
        self._warmup = 0
        self._prices: list[float] = []
        super().__init__()

    # ------------------------------------------------------------------
    # Tasks / reset
    # ------------------------------------------------------------------

    def tasks(self) -> list[Task]:
        return _tasks()

    def reset(self, task: Task) -> str:
        prices, warmup = make_prices(seed=task.metadata["seed"], n_visible=task.metadata["n_visible"])
        self._prices = prices
        self._warmup = warmup
        self._ex = Exchange(prices=prices, cursor=warmup, start_cash=START_CASH)
        snap = self._ex.snapshot()
        return (
            f"State: {snap}. There are six tools available; you do not yet know what they "
            "do. Some change the state when called."
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _build_tools(self) -> None:
        r = self.registry
        r.register(ToolSpec(name="get_price", description="Return the current market price.",
                            parameters={"type": "object", "properties": {}}), self._get_price)
        r.register(ToolSpec(name="get_history", description="Return the last n prices.",
                            parameters={"type": "object",
                                        "properties": {"n": {"type": "integer", "description": "how many recent prices"}}}),
                   self._get_history)
        r.register(ToolSpec(name="advance", description="Let time pass by n steps; returns the new price.",
                            parameters={"type": "object",
                                        "properties": {"steps": {"type": "integer"}}}), self._advance)
        r.register(ToolSpec(name="buy", description="Buy `quantity` units at the current price.",
                            parameters={"type": "object",
                                        "properties": {"quantity": {"type": "number"}}, "required": ["quantity"]}),
                   self._buy)
        r.register(ToolSpec(name="sell", description="Sell `quantity` units at the current price.",
                            parameters={"type": "object",
                                        "properties": {"quantity": {"type": "number"}}, "required": ["quantity"]}),
                   self._sell)
        r.register(ToolSpec(name="portfolio", description="Return cash, position, price, and portfolio_value.",
                            parameters={"type": "object", "properties": {}}), self._portfolio)

    def _ex_or_raise(self) -> Exchange:
        if self._ex is None:
            raise RuntimeError("environment not reset")
        return self._ex

    def _get_price(self) -> float:
        return self._ex_or_raise().price()

    def _get_history(self, n: int = 5) -> list[float]:
        return self._ex_or_raise().history(n)

    def _advance(self, steps: int = 1) -> dict:
        ex = self._ex_or_raise()
        ex.advance(steps)
        return ex.snapshot()

    def _buy(self, quantity: float) -> dict:
        return self._ex_or_raise().buy(quantity)

    def _sell(self, quantity: float) -> dict:
        return self._ex_or_raise().sell(quantity)

    def _portfolio(self) -> dict:
        return self._ex_or_raise().snapshot()

    # ------------------------------------------------------------------
    # Scoring (objective)
    # ------------------------------------------------------------------

    def score(self, task: Task, trajectory: Trajectory) -> TaskResult:
        ex = self._ex_or_raise()
        agent_return = (ex.final_value() - START_CASH) / START_CASH
        oracle = oracle_return(self._prices, self._warmup, START_CASH)

        # score = agent return as a fraction of the clairvoyant optimum, floored at 0.
        score = 0.0
        if oracle > 1e-6 and agent_return > 0:
            score = max(0.0, min(1.0, agent_return / oracle))

        return TaskResult(
            task_id=task.id,
            score=score,
            detail={
                "agent_return": round(agent_return, 4),
                "oracle_return": round(oracle, 4),
                "final_value": round(ex.final_value(), 2),
                "reached_final_step": ex.at_end(),
            },
            trajectory=trajectory,
        )

    # Convenience for tests / scripted strategies ------------------------
    @property
    def exchange(self) -> Exchange:
        return self._ex_or_raise()
