"""Generic tools an environment may expose for the agent to discover.

These are deliberately *real* (they actually run code / hit the network) because the whole
point of the rebuild is that the agent can reach beyond its own training data. Real
network + code execution is exactly why the brief mandates Docker — these are intended to
run inside the container, not on a bare host.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Callable

import httpx

from ..models import ToolSpec

ToolPair = tuple[ToolSpec, Callable[..., Any]]


def make_run_python_tool(timeout: float = 10.0) -> ToolPair:
    """Execute a short Python snippet and capture its output.

    Isolation note: this runs a subprocess with a timeout. That is *not* a security
    sandbox on its own — the Docker container is the actual isolation boundary.
    """

    spec = ToolSpec(
        name="run_python",
        description="Execute a Python 3 snippet and return its stdout/stderr. "
        "Use print() to see values. Runs with a short timeout.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to run"}},
            "required": ["code"],
        },
    )

    def handler(code: str) -> str:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"(timed out after {timeout}s)"
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return out.strip()[:4000] or "(no output)"

    return spec, handler


def make_web_fetch_tool(timeout: float = 15.0, max_chars: int = 6000) -> ToolPair:
    """Fetch a URL and return its text content (HTML crudely stripped to text)."""

    spec = ToolSpec(
        name="web_fetch",
        description="Fetch a URL and return its readable text content.",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Absolute http(s) URL"}},
            "required": ["url"],
        },
    )

    def handler(url: str) -> str:
        if not re.match(r"^https?://", url):
            raise ValueError("url must start with http:// or https://")
        resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": "stem-agent/1.0"})
        resp.raise_for_status()
        text = resp.text
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    return spec, handler


def make_web_search_tool(timeout: float = 15.0, max_results: int = 5) -> ToolPair:
    """Best-effort web search via the DuckDuckGo HTML endpoint.

    Degrades gracefully (returns an explanatory string) if the network is unavailable, so
    a research rollout never crashes just because search is down.
    """

    spec = ToolSpec(
        name="web_search",
        description="Search the web and return a list of result titles + URLs + snippets.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    def handler(query: str) -> Any:
        try:
            resp = httpx.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 stem-agent/1.0"},
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return f"(web_search unavailable: {type(exc).__name__}: {exc})"

        results = []
        for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', resp.text
        ):
            url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
            results.append({"title": title, "url": url})
            if len(results) >= max_results:
                break
        return results or "(no results)"

    return spec, handler
