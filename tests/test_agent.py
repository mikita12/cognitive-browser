"""
Unit tests for AgentCore and AgentOrchestrator using mocks for Playwright and NLI router.
Tests run offline without GPU and without network access.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from cognitive_browser.agent_core import AgentCore
from cognitive_browser.agent_orchestrator import AgentOrchestrator


class TestAgentCoreAndOrchestrator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a mock router
        self.mock_router = MagicMock()
        self.sample_snapshot = """
- banner:
  - searchbox "Search items"
  - button "Submit"
  - link "Help"
"""

    def _create_mock_page(self, snapshot_content: str):
        page = AsyncMock()
        page.aria_snapshot = AsyncMock(return_value=snapshot_content)
        page.goto = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.url = "https://example.com/search"
        page.title = AsyncMock(return_value="Search Results")

        locator = AsyncMock()
        locator.is_visible = AsyncMock(return_value=True)
        locator.click = AsyncMock()
        locator.fill = AsyncMock()
        first_loc = AsyncMock()
        first_loc.click = AsyncMock()
        first_loc.fill = AsyncMock()
        first_loc.is_visible = AsyncMock(return_value=True)
        locator.first = first_loc

        page.locator = MagicMock(return_value=locator)
        page.get_by_role = MagicMock(return_value=locator)
        page.get_by_text = MagicMock(return_value=locator)
        return page

    async def test_high_confidence_step_executes_without_escalation(self):
        self.mock_router.evaluate_candidates.return_value = {
            "target_ref": "aria-1",
            "confidence": 0.92,
            "tool_action": "click()",
            "should_escalate": False,
            "latency_ms": 15.0,
            "ranked_candidates": [{"role": "button", "name": "Submit", "ref": "aria-1"}],
        }

        agent = AgentCore(router=self.mock_router, confidence_threshold=0.50)
        page = self._create_mock_page(self.sample_snapshot)

        res = await agent.execute_step(page=page, instruction="click submit button")
        self.assertTrue(res["executed"])
        self.assertIsNone(res["error"])
        self.assertEqual(res["target_ref"], "aria-1")
        page.get_by_role.assert_called()

    async def test_low_confidence_step_triggers_escalation_flag(self):
        self.mock_router.evaluate_candidates.return_value = {
            "target_ref": "NONE",
            "confidence": 0.28,
            "tool_action": "escalate",
            "should_escalate": True,
            "latency_ms": 12.0,
            "ranked_candidates": [],
        }

        agent = AgentCore(router=self.mock_router, confidence_threshold=0.50)
        page = self._create_mock_page(self.sample_snapshot)

        res = await agent.execute_step(page=page, instruction="click non-existent button")
        self.assertFalse(res["executed"])
        self.assertIn("escalation to Cortex required", res["error"])

    async def test_custom_confidence_threshold_enforcement(self):
        # Confidence is 0.65, which is above 0.50 default but below custom 0.80 threshold
        def fake_eval(intent, candidates, confidence_threshold=0.50):
            conf = 0.65
            return {
                "target_ref": "aria-1" if conf >= confidence_threshold else "NONE",
                "confidence": conf,
                "tool_action": "click",
                "should_escalate": conf < confidence_threshold,
                "latency_ms": 10.0,
                "ranked_candidates": [{"role": "button", "name": "Submit", "ref": "aria-1"}],
            }

        self.mock_router.evaluate_candidates.side_effect = fake_eval

        # With 0.80 threshold, it must flag escalation
        agent = AgentCore(router=self.mock_router, confidence_threshold=0.80)
        page = self._create_mock_page(self.sample_snapshot)

        res = await agent.execute_step(page=page, instruction="click submit")
        self.assertFalse(res["executed"])
        self.assertIn("escalation to Cortex required", res["error"])

    async def test_fill_value_on_non_input_element_returns_error(self):
        self.mock_router.evaluate_candidates.return_value = {
            "target_ref": "aria-1",
            "confidence": 0.90,
            "tool_action": "click",
            "should_escalate": False,
            "latency_ms": 10.0,
            "ranked_candidates": [{"role": "button", "name": "Submit", "ref": "aria-1"}],
        }

        agent = AgentCore(router=self.mock_router)
        page = self._create_mock_page(self.sample_snapshot)

        res = await agent.execute_step(page=page, instruction="type query", fill_value="test text")
        self.assertFalse(res["executed"])
        self.assertIn("does not accept text input", res["error"])

    async def test_orchestrator_recovers_via_cortex_escalation(self):
        # Step 1: Reflex fails
        self.mock_router.evaluate_candidates.return_value = {
            "target_ref": "NONE",
            "confidence": 0.35,
            "tool_action": "escalate",
            "should_escalate": True,
            "latency_ms": 20.0,
            "ranked_candidates": [],
        }

        orchestrator = AgentOrchestrator(router=self.mock_router)
        page = self._create_mock_page(self.sample_snapshot)

        with patch.object(
            orchestrator,
            "call_cortex_llm",
            return_value={
                "chosen_ref": "aria-1",
                "chosen_name": "Submit",
                "action": "click",
                "tokens": 180,
                "latency_ms": 150.0,
            },
        ):
            task_res = await orchestrator.execute_task(
                page=page, url="https://example.com/checkout", steps=[{"instruction": "Submit checkout form"}]
            )

        self.assertEqual(task_res["status"], "completed")
        self.assertEqual(task_res["escalations_count"], 1)
        self.assertEqual(task_res["cortex_tokens"], 180)
        self.assertTrue(task_res["steps_log"][-1]["assisted_by_cortex"])

    def test_cortex_llm_fallback_on_network_failure(self):
        orchestrator = AgentOrchestrator(router=self.mock_router)
        with patch("requests.post", side_effect=Exception("Connection refused")):
            candidates = [{"role": "button", "name": "OK", "ref": "aria-1"}]
            res = orchestrator.call_cortex_llm(intent="click OK", candidates=candidates, aria_tree="")
            self.assertEqual(res["chosen_ref"], "aria-1")
            self.assertIn("error", res)
            self.assertIn("Connection refused", res["error"])


if __name__ == "__main__":
    unittest.main()
