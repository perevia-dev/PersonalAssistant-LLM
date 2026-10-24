"""Bounded, permission-aware Planner action loop.

A small independent implementation of Maisaka's key *execution semantics*:
plan -> authorized action -> feedback -> replan -> reply/wait/stop.
No channel I/O, model SDK, filesystem, or plugins are imported here.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace as dataclass_replace
from time import monotonic
from typing import Awaitable, Callable, Literal

from .planner import PlanDecision, choose_action


@dataclass(frozen=True, slots=True)
class ToolReceipt:
    name: str
    success: bool
    summary: str

    def as_context(self) -> str:
        # Tool output is untrusted data. Never embed arbitrary tool results in
        # a system instruction or assume an action succeeded without a receipt.
        safe = " ".join(self.summary.split())[:450]
        return f"{self.name}: {'成功' if self.success else '失败'}；{safe}"


@dataclass(frozen=True, slots=True)
class ToolBinding:
    name: str
    handler: Callable[[dict], Awaitable[str]]
    scopes: frozenset[str] = field(default_factory=lambda: frozenset({"private"}))
    mutating: bool = False
    description: str = ""
    parameters: dict = field(default_factory=dict)
    visibility: Literal["core", "deferred"] = "core"
    discovery_terms: tuple[str, ...] = ()


class AuthorizedToolRegistry:
    """Explicit host registration and per-call authorization; deny by default."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolBinding] = {}

    def register(self, tool: ToolBinding) -> None:
        self.register_many((tool,))

    def register_many(self, bindings: tuple[ToolBinding, ...] | list[ToolBinding]) -> None:
        """All-or-nothing registration; a failed skill cannot install half its tools."""
        names = [tool.name for tool in bindings]
        if not names or any(not name or name in self._tools for name in names):
            raise ValueError("Tool name empty or already registered")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate tool names within one skill")
        if any(not callable(tool.handler) for tool in bindings):
            raise ValueError("Tool handler must be callable")
        self._tools.update({tool.name: tool for tool in bindings})

    def available(self, name: str, *, scope: str, allow_mutation: bool = False) -> bool:
        tool = self._tools.get(name)
        return bool(tool and scope in tool.scopes and (not tool.mutating or allow_mutation))

    def public_names(self, *, scope: str, allow_mutation: bool = False) -> tuple[str, ...]:
        return tuple(name for name in self._tools if self.available(name, scope=scope, allow_mutation=allow_mutation))

    def discovery_bindings(self, *, scope: str,
                           allow_mutation: bool = False) -> tuple[ToolBinding, ...]:
        return tuple(binding for binding in self._tools.values()
                     if self.available(binding.name, scope=scope,
                                       allow_mutation=allow_mutation))

    def prompt_catalog(self, *, scope: str, allow_mutation: bool = False,
                       names: tuple[str, ...] | None = None) -> str:
        """Only host-registered, authorized tool schemas are shown to the model."""
        import json
        lines = []
        allowed = self.public_names(scope=scope, allow_mutation=allow_mutation)
        chosen = allowed[:12] if names is None else tuple(
            name for name in names if name in allowed)[:12]
        for name in chosen:
            tool = self._tools[name]
            spec = json.dumps(tool.parameters, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"{name}：{tool.description[:150]}；参数 schema：{spec[:450]}")
        return "\n".join(lines)[:2400]

    @staticmethod
    def _check_schema(arguments: dict, schema: dict) -> bool:
        """Bounded host-side JSON Schema subset. Unknown keywords fail closed.

        Supports objects, arrays, enums, primitives, required, bounds and
        additionalProperties=false; never interprets JSON references.
        """
        import math
        allowed_keys = frozenset({
            "type", "properties", "required", "additionalProperties",
            "items", "enum", "minLength", "maxLength",
            "minimum", "maximum", "minItems", "maxItems",
            "description",
        })

        def validate(value, spec, depth=0):
            if depth > 6 or not isinstance(spec, dict) or set(spec) - allowed_keys:
                return False
            if not spec:
                return True
            kind = spec.get("type")
            kinds = {
                "object": lambda: type(value) is dict,
                "array": lambda: type(value) is list,
                "string": lambda: type(value) is str,
                "integer": lambda: type(value) is int,
                "number": lambda: type(value) in (int, float)
                and math.isfinite(value),
                "boolean": lambda: type(value) is bool,
            }
            if kind not in kinds or not kinds[kind]():
                return False
            if "enum" in spec:
                options = spec["enum"]
                if not isinstance(options, list) or not options or value not in options:
                    return False
            if kind == "object":
                props = spec.get("properties", {})
                required = spec.get("required", [])
                extra = spec.get("additionalProperties", True)
                if (not isinstance(props, dict) or not isinstance(required, list)
                        or any(not isinstance(x, str) for x in required)
                        or extra not in (True, False)):
                    return False
                if any(key not in value for key in required):
                    return False
                if extra is False and any(key not in props for key in value):
                    return False
                return all(validate(v, props[k], depth+1)
                           for k, v in value.items() if k in props)
            if kind == "array":
                items = spec.get("items")
                low, high = spec.get("minItems", 0), spec.get("maxItems", 100)
                if (type(low) is not int or type(high) is not int
                        or not 0 <= low <= len(value) <= high <= 100):
                    return False
                return items is None or all(validate(v, items, depth+1) for v in value)
            if kind == "string":
                low, high = spec.get("minLength", 0), spec.get("maxLength", 1024)
                return (type(low) is int and type(high) is int
                        and 0 <= low <= len(value) <= high <= 3000)
            if kind in ("integer", "number"):
                lower, upper = spec.get("minimum", float("-inf")), spec.get("maximum", float("inf"))
                return (type(lower) in (float, int) and type(upper) in (float, int)
                        and lower <= value <= upper)
            return True

        return type(arguments) is dict and validate(arguments, schema)

    async def invoke(self, name: str, arguments: dict, *, scope: str,
                     allow_mutation: bool = False, timeout: float = 6.0) -> ToolReceipt:
        if not self.available(name, scope=scope, allow_mutation=allow_mutation):
            return ToolReceipt(name, False, "工具未授权或不可用")
        if not isinstance(arguments, dict) or len(repr(arguments)) > 3000:
            return ToolReceipt(name, False, "工具参数不合法或过长")
        if not self._check_schema(arguments, self._tools[name].parameters):
            return ToolReceipt(name, False, "工具参数不符合已注册的 schema")
        try:
            output = await asyncio.wait_for(self._tools[name].handler(dict(arguments)),
                                            timeout=max(0.1, timeout))
            return ToolReceipt(name, True, str(output)[:450])
        except (Exception, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            return ToolReceipt(name, False, f"执行失败（{type(exc).__name__}）")


@dataclass(frozen=True, slots=True)
class CycleResult:
    plan: PlanDecision
    action: Literal["reply", "wait", "stop"]
    receipts: tuple[ToolReceipt, ...]
    rounds: int
    reason: str = ""

    def tool_feedback(self) -> str:
        """Bounded evidence with explicit outcome; preserve complete receipt lines."""
        if not self.receipts:
            return ""
        lines = [
            f"工具循环已结束；原因={self.reason or 'planner_finished'}；"
            f"成功={sum(r.success for r in self.receipts)}；"
            f"失败={sum(not r.success for r in self.receipts)}。",
            "已经执行的工具不得再次表述为即将执行；只根据以下回执回答，"
            "未获得的内容不可编造。",
        ]
        for receipt in self.receipts:
            line = receipt.as_context()
            if len("\n".join(lines + [line])) > 1200:
                lines.append("其余回执因上下文预算限制未展示。")
                break
            lines.append(line)
        return "\n".join(lines)[:1200]



def _search_terms(query: str) -> set[str]:
    """Conservative language-independent anchors for same-turn search checks."""
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", query.casefold())
    terms: set[str] = set()
    for word in words:
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            terms.update(word[i:i+2] for i in range(len(word)-1))
        elif len(word) >= 3:
            terms.add(word)
    return terms


def _search_relation(previous: str, proposed: str) -> str:
    """Detect near-repeat or unrelated query, without claiming full semantics."""
    old, new = _search_terms(previous), _search_terms(proposed)
    if not old or not new:
        return "unknown"
    common = len(old & new)
    overlap = common / min(len(old), len(new))
    similarity = common / len(old | new)
    if similarity >= .72:
        return "duplicate"
    if overlap < .18:
        return "topic_drift"
    return "related"


class AgentCycleEngine:
    """One logical turn. The event scheduler decides whether to create it.

    Tools must be independently registered/authorized. No tool is installed by
    default. A failed/denied tool cannot silently become a successful action.
    """

    def __init__(self, *, max_rounds: int = 4, max_tools: int = 2,
                 total_timeout: float = 30.0) -> None:
        if not 1 <= max_rounds <= 8 or not 0 <= max_tools <= 6 or not 1 <= total_timeout <= 120:
            raise ValueError("Agent cycle budgets outside allowed bounds")
        self.max_rounds = max_rounds
        self.max_tools = max_tools
        self.total_timeout = total_timeout

    async def run(
        self,
        planner: Callable[[tuple[ToolReceipt, ...]], Awaitable[PlanDecision]],
        *,
        scope: str = "private",
        registry: AuthorizedToolRegistry | None = None,
        allow_tools: bool = False,
        allow_mutation: bool = False,
        allow_silence: bool = False,
        already_waited: bool = False,
        tool_timeout: float = 6.0,
        on_event: Callable[[str, int, str], None] | None = None,
    ) -> CycleResult:
        deadline = monotonic() + self.total_timeout
        receipts: list[ToolReceipt] = []
        invoked: set[tuple[str, str]] = set()
        tools_used = 0
        previous_searches: list[str] = []
        fallback = PlanDecision()

        def emit(event: str, round_no: int, detail: str = "") -> None:
            if on_event is not None:
                try:
                    on_event(event, round_no, detail)
                except Exception:
                    pass  # Diagnostics must never change action execution.

        def finalize(plan: PlanDecision, reason: str, round_no: int) -> CycleResult:
            # A request to use a tool is never a valid final reply instruction.
            # Keep the character/tone, but clear stale tool intent and arguments.
            if plan.action == "tool":
                plan = dataclass_replace(plan, action="reply",
                    reply_act="依据已经执行的工具回执回答当前问题；不要称正在准备查询",
                    tool_name="", tool_arguments=None, selection_reason="")
            emit("finalize", round_no, reason)
            return CycleResult(plan, "reply", tuple(receipts), round_no, reason)

        for round_no in range(1, self.max_rounds + 1):
            remaining = deadline - monotonic()
            if remaining <= 0:
                return finalize(fallback, "total_timeout", round_no-1)
            try:
                plan = await asyncio.wait_for(planner(tuple(receipts)), timeout=remaining)
            except asyncio.CancelledError:
                raise
            except Exception:
                return finalize(fallback, "planner_failure", round_no)
            if not isinstance(plan, PlanDecision):
                return finalize(fallback, "invalid_plan", round_no)
            fallback = plan
            emit("plan", round_no, plan.action if plan.action != "tool" else f"tool:{plan.tool_name}")
            authorized = bool(
                allow_tools and registry is not None
                and registry.available(plan.tool_name, scope=scope, allow_mutation=allow_mutation)
            )
            action = choose_action(plan, allow_silence=allow_silence,
                                   already_waited=already_waited, tools_enabled=authorized)
            if action != "tool":
                emit("action", round_no, action if action == plan.action else f"{plan.action}->{action}")
                reason = "tool_denied" if plan.action == "tool" else ""
                return CycleResult(plan, action, tuple(receipts), round_no, reason)
            if plan.tool_name == "web_search.search" and previous_searches:
                query = (plan.tool_arguments or {}).get("query", "")
                if isinstance(query, str):
                    relations = [_search_relation(q, query) for q in previous_searches]
                    if "duplicate" in relations:
                        return finalize(plan, "duplicate_search", round_no)
                    if "related" not in relations and "unknown" not in relations:
                        return finalize(plan, "search_topic_drift", round_no)
            if tools_used >= self.max_tools or round_no >= self.max_rounds:
                return finalize(plan, "budget_exhausted", round_no)
            key = (plan.tool_name, repr(sorted((plan.tool_arguments or {}).items())))
            if key in invoked:
                return finalize(plan, "duplicate_tool", round_no)
            invoked.add(key)
            tools_used += 1
            remaining = deadline - monotonic()
            if remaining <= 0:
                return finalize(plan, "total_timeout", round_no)
            emit("tool_start", round_no, plan.tool_name)
            receipt = await registry.invoke(
                plan.tool_name, plan.tool_arguments or {},
                scope=scope, allow_mutation=allow_mutation,
                timeout=min(tool_timeout, remaining),
            )
            receipts.append(receipt)
            emit("tool_end", round_no, f"{plan.tool_name}:{'success' if receipt.success else 'failure'}")
            if receipt.success and plan.tool_name == "web_search.search":
                query = (plan.tool_arguments or {}).get("query", "")
                if isinstance(query, str):
                    previous_searches.append(query)
            # Failed tools do not trigger repeated retries or pretend success.
            if not receipt.success:
                return finalize(plan, "tool_failed", round_no)
        return finalize(fallback, "budget_exhausted", self.max_rounds)
