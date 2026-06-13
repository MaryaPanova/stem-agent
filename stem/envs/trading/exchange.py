"""A deterministic simulated exchange.

Fully reproducible from a seed. The price series is a noisy sinusoid that completes whole
cycles over the episode, so:

  * **buy-and-hold** ends roughly where it started  (~0 return)
  * **do nothing**   ends with exactly the starting cash (0 return)
  * **buy troughs / sell peaks** captures the oscillation (clearly positive return)

That spread is what makes a genuine ``~0`` baseline possible: a fresh agent that never
figures out the trade-the-cycle pattern simply does not make money, while an evolved one
that discovers it does.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def make_prices(
    seed: int,
    n_visible: int = 24,
    warmup: int = 6,
    base: float = 100.0,
    amplitude: float = 12.0,
    cycles: float = 2.0,
    noise: float = 0.5,
) -> tuple[list[float], int]:
    """Return (prices, warmup_index). ``cursor`` starts at ``warmup`` so history exists."""
    total = warmup + n_visible
    rng = random.Random(seed)
    prices: list[float] = []
    for i in range(total):
        phase = 2 * math.pi * cycles * (i / (total - 1))
        p = base + amplitude * math.sin(phase) + rng.uniform(-noise, noise)
        prices.append(round(p, 2))
    return prices, warmup


@dataclass
class Exchange:
    prices: list[float]
    cursor: int = 0
    start_cash: float = 1000.0
    cash: float = field(default=0.0)
    position: float = 0.0

    def __post_init__(self) -> None:
        self.cash = self.start_cash

    # --- market ---------------------------------------------------------
    def price(self) -> float:
        return self.prices[self.cursor]

    def history(self, n: int = 5) -> list[float]:
        n = max(1, int(n))
        return self.prices[max(0, self.cursor - n + 1) : self.cursor + 1]

    def at_end(self) -> bool:
        return self.cursor >= len(self.prices) - 1

    def advance(self, steps: int = 1) -> float:
        self.cursor = min(self.cursor + max(1, int(steps)), len(self.prices) - 1)
        return self.price()

    # --- trading --------------------------------------------------------
    def buy(self, quantity: float) -> dict:
        quantity = float(quantity)
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        cost = quantity * self.price()
        if cost > self.cash + 1e-9:
            raise ValueError(f"insufficient cash: need {cost:.2f}, have {self.cash:.2f}")
        self.cash -= cost
        self.position += quantity
        return self.snapshot()

    def sell(self, quantity: float) -> dict:
        quantity = float(quantity)
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self.position + 1e-9:
            raise ValueError(f"insufficient position: have {self.position:.4f}")
        self.cash += quantity * self.price()
        self.position -= quantity
        return self.snapshot()

    # --- accounting -----------------------------------------------------
    def value(self) -> float:
        return self.cash + self.position * self.price()

    def snapshot(self) -> dict:
        return {
            "cash": round(self.cash, 2),
            "position": round(self.position, 4),
            "price": self.price(),
            "portfolio_value": round(self.value(), 2),
            "step": self.cursor,
            "final_step": len(self.prices) - 1,
        }

    def final_value(self) -> float:
        """Mark-to-market value if everything were liquidated at the current price."""
        return self.value()


def oracle_return(prices: list[float], warmup: int, start_cash: float = 1000.0) -> float:
    """Return of a clairvoyant buy-troughs / sell-peaks strategy over the full series.

    Used as the denominator when scoring, so a perfect play scores ~1.0. All-in / all-out
    at every local extremum from ``warmup`` to the end.
    """
    cash, position = start_cash, 0.0
    segment = prices[warmup:]
    for i in range(1, len(segment) - 1):
        prev, cur, nxt = segment[i - 1], segment[i], segment[i + 1]
        is_trough = cur <= prev and cur < nxt
        is_peak = cur >= prev and cur > nxt
        if is_trough and cash > 0:
            position = cash / cur
            cash = 0.0
        elif is_peak and position > 0:
            cash = position * cur
            position = 0.0
    final = cash + position * segment[-1]
    return (final - start_cash) / start_cash
