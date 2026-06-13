"""Deterministic exchange + objective trading score."""

from stem.envs.trading.env import START_CASH, TradingEnvironment
from stem.envs.trading.exchange import Exchange, make_prices, oracle_return
from stem.models import Trajectory


def test_prices_are_deterministic():
    a, wa = make_prices(seed=11, n_visible=24)
    b, wb = make_prices(seed=11, n_visible=24)
    assert a == b and wa == wb
    assert make_prices(seed=12, n_visible=24)[0] != a


def test_buy_sell_accounting():
    ex = Exchange(prices=[100.0, 110.0], cursor=0, start_cash=1000.0)
    ex.buy(5)
    assert ex.cash == 500.0 and ex.position == 5
    ex.advance()
    assert ex.price() == 110.0
    ex.sell(5)
    assert ex.cash == 1050.0 and ex.position == 0
    assert ex.value() == 1050.0


def test_cannot_overspend_or_oversell():
    ex = Exchange(prices=[100.0], cursor=0, start_cash=50.0)
    for bad in (lambda: ex.buy(1), lambda: ex.sell(1), lambda: ex.buy(-1)):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_oracle_beats_do_nothing_and_buy_hold():
    prices, warm = make_prices(seed=11, n_visible=24)
    seg = prices[warm:]
    buy_hold = (seg[-1] - seg[0]) / seg[0]
    oracle = oracle_return(prices, warm)
    assert oracle > 0.2          # real headroom
    assert buy_hold < oracle     # passive strategy leaves money on the table


def test_do_nothing_scores_zero():
    env = TradingEnvironment()
    task = env.test_tasks()[0]
    env.reset(task)
    result = env.score(task, Trajectory(task_id=task.id))
    assert result.score == 0.0
    assert result.detail["agent_return"] == 0.0


def test_profitable_trades_score_positive():
    env = TradingEnvironment()
    task = next(t for t in env.tasks() if t.id == "trade-train-11")
    prices, warm = make_prices(**{"seed": 11, "n_visible": task.metadata["n_visible"]})
    seg = prices[warm:]
    # buy at the segment minimum, sell at the maximum that follows it
    lo = min(range(len(seg)), key=lambda i: seg[i])
    hi = max(range(lo + 1, len(seg)), key=lambda i: seg[i])

    env.reset(task)
    if lo > 0:
        env.execute("advance", {"steps": lo})
    env.execute("buy", {"quantity": 1.0})
    env.execute("advance", {"steps": hi - lo})
    env.execute("sell", {"quantity": 1.0})

    result = env.score(task, Trajectory(task_id=task.id))
    assert result.detail["agent_return"] > 0
    assert result.score > 0
