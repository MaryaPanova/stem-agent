"""Lookup table from domain name -> Environment factory.

Adding a domain is adding one line here. The agent, evolver, and eval harness are all
domain-agnostic, so nothing else changes.
"""

from __future__ import annotations

from typing import Callable

from ..environment import Environment


def _trading() -> Environment:
    from .trading import TradingEnvironment

    return TradingEnvironment()


def _security() -> Environment:
    from .security import SecurityEnvironment

    return SecurityEnvironment()


def _research() -> Environment:
    from .research import ResearchEnvironment

    return ResearchEnvironment()


_REGISTRY: dict[str, Callable[[], Environment]] = {
    "trading": _trading,
    "security": _security,
    "research": _research,
}


def list_environments() -> list[str]:
    return list(_REGISTRY)


def get_environment(name: str) -> Environment:
    if name not in _REGISTRY:
        raise KeyError(f"unknown environment {name!r}; available: {list_environments()}")
    return _REGISTRY[name]()
