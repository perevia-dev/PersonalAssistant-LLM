"""Offline Agent V2 scheduler, tools and event admission regressions."""
import unittest

from zhuangfy.agent_v2.scheduler import SchedulerConfig, UnifiedTurnScheduler
from zhuangfy.agent_v2.turn_trigger import TurnEvent
from zhuangfy.agent_v2.tools import build_builtin_registry
from zhuangfy.agent_v2.runtime import AgentCycleEngine, AuthorizedToolRegistry, ToolBinding, _search_relation
from zhuangfy.agent_v2.planner import PlanDecision


def group_event(mid="1", directed=False):
    return TurnEvent("qq:bot:group:9", mid, "message", "group",
                     text="今天要不要继续聊下去？", directed=directed)


class SchedulerTests(unittest.TestCase):
    def test_directed_message_and_wait_continuation(self):
        scheduler = UnifiedTurnScheduler(SchedulerConfig(group_frequency=0.0))
        self.assertEqual(scheduler.decide(group_event("a", True), now=1).decision, "trigger")
        self.assertEqual(scheduler.decide(group_event("a", True), now=2).decision, "ignore")
        self.assertEqual(scheduler.decide(group_event("b"), now=3).decision, "hold")
        self.assertEqual(scheduler.decide(TurnEvent("qq:bot:private:42", "p", "message",
                    "private", text="你好"), now=3).decision, "trigger")
        self.assertEqual(scheduler.decide(TurnEvent("qq:bot:group:9", "tool", "tool_result",
                    "group", parent_turn_active=True), now=3).decision, "trigger")

    def test_group_frequency_and_backoff(self):
        scheduler = UnifiedTurnScheduler(SchedulerConfig(group_frequency=1.0,
                    idle_backoff_start=1, idle_backoff_base_seconds=2.0))
        self.assertEqual(scheduler.decide(group_event("a"), now=100).decision, "trigger")
        scheduler.record_cycle("qq:bot:group:9", now=100, action="stop", group=True)
        self.assertEqual(scheduler.decide(group_event("b"), now=101).decision, "delay")
        self.assertEqual(scheduler.decide(group_event("b"), now=103).decision, "trigger")
        scheduler.record_cycle("qq:bot:group:9", now=103, action="reply", delivered=True, group=True)
        self.assertEqual(scheduler.metrics("qq:bot:group:9", now=104)["actual_replies"], 1.0)

    def test_debounce_precedes_admission_and_permission(self):
        scheduler = UnifiedTurnScheduler()
        self.assertEqual(scheduler.decide(group_event("a", True), now=1,
                                         debounce_remaining=2).decision, "delay")
        self.assertEqual(scheduler.decide(group_event("a", True), now=3).decision, "trigger")
        self.assertEqual(scheduler.decide(TurnEvent("qq:bot:group:9", "x",
                        "message", "group", authorized=False), now=3).decision, "ignore")
        with self.assertRaises(ValueError):
            SchedulerConfig(group_frequency=float("inf"))


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_builtins_return_verified_receipts(self):
        reg = build_builtin_registry()
        self.assertEqual(set(reg.public_names(scope="group")),
                         {"calculator.evaluate", "clock.utc_now"})
        result = await reg.invoke("calculator.evaluate", {"expression":"2*(3+4)/7"},
                                  scope="group")
        self.assertTrue(result.success)
        self.assertIn("2", result.summary)
        clock = await reg.invoke("clock.utc_now", {}, scope="private")
        self.assertTrue(clock.success)
        self.assertIn("UTC", clock.summary)

    async def test_unknown_and_unsafe_expressions_are_rejected(self):
        reg = build_builtin_registry()
        for expression in ["__import__('os').system('echo hi')", "2**100000",
                           "[1,2]", "1/0", "1e300"]:
            with self.subTest(expression=expression):
                result = await reg.invoke("calculator.evaluate",
                                          {"expression":expression}, scope="private")
                self.assertFalse(result.success)
        denied = await reg.invoke("mijia.turn_on", {"device":"x"}, scope="private")
        self.assertFalse(denied.success)

    async def test_live_planner_tool_then_reply(self):
        reg = build_builtin_registry()
        steps = []
        async def planner(receipts):
            steps.append(tuple(receipts))
            if not receipts:
                return PlanDecision(action="tool", confidence=.95,
                                    tool_name="calculator.evaluate",
                                    tool_arguments={"expression":"10+15"})
            return PlanDecision(action="reply", confidence=.95,
                                reply_act="告知真实计算结果")
        result = await AgentCycleEngine().run(planner, registry=reg,
                                              allow_tools=True, scope="private")
        self.assertEqual(result.action, "reply")
        self.assertEqual(result.rounds, 2)
        self.assertIn("25", result.tool_feedback())
        self.assertEqual(len(steps), 2)


    async def test_budget_exhausted_preserves_receipt_and_finalizes_tool_plan(self):
        reg = build_builtin_registry()
        async def planner(receipts):
            return PlanDecision(action="tool", confidence=.95,
                                tool_name="calculator.evaluate",
                                tool_arguments={"expression": "10+15" if not receipts else "8+9"},
                                social_intent="回答计算问题")
        result = await AgentCycleEngine(max_tools=1, max_rounds=3).run(
            planner, registry=reg, allow_tools=True)
        self.assertEqual(result.reason, "budget_exhausted")
        self.assertEqual(result.action, "reply")
        self.assertEqual(result.plan.action, "reply")
        self.assertEqual(result.plan.tool_name, "")
        self.assertEqual(len(result.receipts), 1)
        self.assertIn("25", result.tool_feedback())
        self.assertIn("已经执行", result.tool_feedback())

    async def test_duplicate_tool_finalizes_without_second_execution(self):
        reg = build_builtin_registry()
        async def planner(receipts):
            return PlanDecision(action="tool", confidence=.95,
                                tool_name="calculator.evaluate",
                                tool_arguments={"expression": "2+3"})
        result = await AgentCycleEngine(max_tools=3, max_rounds=4).run(
            planner, registry=reg, allow_tools=True)
        self.assertEqual(result.reason, "duplicate_tool")
        self.assertEqual(result.plan.action, "reply")
        self.assertEqual(len(result.receipts), 1)
        self.assertIn("5", result.tool_feedback())

    async def test_failed_tool_finalizes_as_reply_with_failure_evidence(self):
        reg = build_builtin_registry()
        async def planner(receipts):
            return PlanDecision(action="tool", confidence=.95,
                                tool_name="calculator.evaluate",
                                tool_arguments={"expression": "1/0"})
        result = await AgentCycleEngine().run(planner, registry=reg,
                                              allow_tools=True)
        self.assertEqual(result.reason, "tool_failed")
        self.assertEqual(result.plan.action, "reply")
        self.assertIn("失败=1", result.tool_feedback())

    async def test_normal_reply_without_tool_keeps_original_intent(self):
        async def planner(receipts):
            return PlanDecision(action="reply", confidence=.9,
                                reply_act="自然回复")
        result = await AgentCycleEngine().run(planner)
        self.assertEqual(result.plan.reply_act, "自然回复")
        self.assertEqual(result.tool_feedback(), "")


    async def test_search_repetition_and_topic_drift_do_not_execute(self):
        for second, reason in (
            ("赛博朋克 边缘行者 第二季 剧情 角色", "duplicate_search"),
            ("武陵 天气 文化活动 近期", "search_topic_drift"),
        ):
            with self.subTest(query=second):
                calls = []
                async def fake_search(args):
                    calls.append(args["query"])
                    return "已获得公开 HTTPS 搜索摘要"
                registry = AuthorizedToolRegistry()
                registry.register(ToolBinding(
                    name="web_search.search", handler=fake_search,
                    scopes=frozenset({"private"}),
                    parameters={"type":"object","properties":{"query":{"type":"string"}},
                                "required":["query"],"additionalProperties":False}))
                async def planner(receipts):
                    return PlanDecision(
                        action="tool", confidence=.95, tool_name="web_search.search",
                        tool_arguments={"query": "赛博朋克 边缘行者 第二季 剧情" if not receipts else second})
                result = await AgentCycleEngine(max_tools=3).run(
                    planner, registry=registry, allow_tools=True)
                self.assertEqual(result.reason, reason)
                self.assertEqual(len(calls), 1)
                self.assertEqual(result.plan.action, "reply")
                self.assertEqual(len(result.receipts), 1)

    async def test_related_novel_search_is_allowed(self):
        self.assertEqual(_search_relation(
            "赛博朋克 边缘行者 第二季 剧情",
            "赛博朋克 边缘行者 第二季 Netflix 官方人物介绍"), "related")


    async def test_current_turn_tool_evidence_is_not_old_history(self):
        from zhuangfy.llm.providers import build_tool_evidence_message
        receipt = "web_search.search: 成功；[1] Roman review | https://example.org/roman"
        message = build_tool_evidence_message(receipt)
        self.assertEqual(message["role"], "user")
        self.assertIn("本轮刚完成", message["content"])
        self.assertIn(receipt, message["content"])
        self.assertNotIn("较早对话", message["content"])
        self.assertIn("证据不足", message["content"])


    async def test_ordered_events_and_stop_prevents_reply(self):
        events = []
        reg = build_builtin_registry()
        async def planner(receipts):
            if not receipts:
                return PlanDecision(action="tool", confidence=.95,
                                    tool_name="calculator.evaluate",
                                    tool_arguments={"expression": "3+4"})
            return PlanDecision(action="stop", confidence=.95)
        result = await AgentCycleEngine().run(
            planner, registry=reg, allow_tools=True, allow_silence=True,
            on_event=lambda event, round_no, detail: events.append((event, round_no, detail)))
        self.assertEqual(result.action, "stop")
        self.assertEqual([x[0] for x in events],
                         ["plan", "tool_start", "tool_end", "plan", "action"])
        self.assertIn("7", result.tool_feedback())

    async def test_private_silence_requires_host_permission(self):
        async def planner(receipts):
            return PlanDecision(action="stop", confidence=.95)
        denied = await AgentCycleEngine().run(planner, allow_silence=False)
        allowed = await AgentCycleEngine().run(planner, allow_silence=True)
        self.assertEqual(denied.action, "reply")
        self.assertEqual(allowed.action, "stop")


if __name__ == "__main__":
    unittest.main()
