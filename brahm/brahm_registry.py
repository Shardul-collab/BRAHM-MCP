"""
brahm/brahm_registry.py
========================
Central tool registry for the BRAHM MCP server.

@brahm_tool   — decorator that registers a handler in one place
registry      — singleton ToolRegistry
@requires_api — guards a handler behind an API availability check
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Coroutine
from mcp import types

log = logging.getLogger("mcp.brahm.registry")


class ToolRegistry:

    def __init__(self) -> None:
        self._tools:    dict[str, types.Tool]               = {}
        self._handlers: dict[str, Callable]                 = {}
        self._groups:   dict[str, list[str]]                = {}

    def register(self, name: str, group: str, description: str,
                 input_schema: dict, handler: Callable) -> None:
        if name in self._tools:
            log.warning("Tool '%s' already registered — overwriting.", name)
        self._tools[name] = types.Tool(
            name        = name,
            description = description,
            inputSchema = input_schema,
        )
        self._handlers[name] = handler
        self._groups.setdefault(group, []).append(name)
        log.debug("Registered tool '%s' in group '%s'", name, group)

    def all_tools(self) -> list[types.Tool]:
        return list(self._tools.values())

    async def dispatch(self, name: str, args: dict) -> dict:
        handler = self._handlers.get(name)
        if handler is None:
            from brahm.shared.helpers import _err
            return _err(f"Unknown tool: {name}")
        return await handler(args)

    def summary(self) -> dict[str, list[str]]:
        return {g: list(names) for g, names in self._groups.items()}

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


registry = ToolRegistry()


def brahm_tool(name: str, group: str, description: str,
               input_schema: dict) -> Callable:
    """
    Decorator that registers a handler with the global registry.

    Usage:
        @brahm_tool(
            name         = "shani_run_workflow",
            group        = "shani",
            description  = "Start a paused workflow...",
            input_schema = {"type": "object", "properties": {...}, "required": [...]},
        )
        async def shani_run_workflow(args: dict) -> dict:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        registry.register(
            name         = name,
            group        = group,
            description  = description,
            input_schema = input_schema,
            handler      = fn,
        )
        return fn
    return decorator


def requires_api(check_fn: Callable[[], Coroutine[Any, Any, bool]],
                 agent_name: str, start_hint: str = "") -> Callable:
    """
    Guards a handler behind an API availability check.
    Apply AFTER @brahm_tool (closer to the function).

    Usage:
        @brahm_tool(...)
        @requires_api(_check_shani, "SHANI", SHANI_START_HINT)
        async def shani_run_workflow(args: dict) -> dict:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        async def wrapper(args: dict) -> dict:
            from brahm.shared.helpers import _err
            if not await check_fn():
                return _err(f"{agent_name} API not running.", start_hint)
            return await fn(args)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__  = fn.__doc__
        return wrapper
    return decorator
