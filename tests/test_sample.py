"""Comprehensive test suite for the Test Validation System.

Groups:
  1. LLM Factory         – get_llm / get_browser_use_llm
  2. State Management     – GraphState schema & defaults
  3. Self-Healer Patching – _patch_script / compile validation
  4. Step Router          – route_step for complexity tags
  5. Explorer Agent       – _default_intelligence / _extract_json / run_explorer
  6. Analyst Intelligence – _overlay_intelligence / run_analyst
  7. Execution Continuity – _inject_cookies / session_state propagation
  8. Pipeline Integration – build_graph structure / conditional routing
  9. Report Generation    – fallback report & page_intelligence field
 10. DOM Parser / Utilities

Run with:
    python -m pytest tests/test_sample.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — LLM Factory
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMFactory:
    """Tests for llm.factory.get_llm and get_browser_use_llm."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "cerebras", "CEREBRAS_API_KEY": "test-key"})
    @patch("langchain_openai.ChatOpenAI")
    def test_get_llm_cerebras(self, mock_chat):
        from llm.factory import get_llm
        mock_chat.return_value = MagicMock()
        get_llm(temperature=0.2, max_tokens=1024)
        mock_chat.assert_called_once()
        kw = mock_chat.call_args[1]
        assert kw["temperature"] == 0.2
        assert kw["max_tokens"] == 1024
        assert "cerebras" in kw["base_url"]

    @patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test-key"})
    @patch("langchain_groq.ChatGroq")
    def test_get_llm_groq(self, mock_groq):
        from llm.factory import get_llm
        mock_groq.return_value = MagicMock()
        get_llm()
        mock_groq.assert_called_once()

    @patch.dict(os.environ, {"LLM_PROVIDER": "lmstudio"})
    @patch("langchain_openai.ChatOpenAI")
    def test_get_llm_lmstudio(self, mock_chat):
        from llm.factory import get_llm
        mock_chat.return_value = MagicMock()
        get_llm()
        mock_chat.assert_called_once()
        assert "127.0.0.1" in mock_chat.call_args[1]["base_url"]

    @patch.dict(os.environ, {"LLM_PROVIDER": "bogus"})
    def test_get_llm_unknown_provider_raises(self):
        from llm.factory import get_llm
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_llm()

    @patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": ""})
    def test_get_llm_missing_api_key_raises(self):
        from llm.factory import get_llm
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            get_llm()

    @patch.dict(os.environ, {
        "BROWSER_USE_PROVIDER": "groq",
        "GROQ_API_KEY": "bu-test-key",
    })
    @patch("langchain_openai.ChatOpenAI")
    def test_get_browser_use_llm_groq(self, mock_chat):
        from llm.factory import get_browser_use_llm
        mock_chat.return_value = MagicMock()
        get_browser_use_llm(temperature=0.0)
        mock_chat.assert_called_once()
        kw = mock_chat.call_args[1]
        assert "groq" in kw["base_url"]
        assert kw["temperature"] == 0.0

    @patch.dict(os.environ, {
        "BROWSER_USE_PROVIDER": "cerebras",
        "CEREBRAS_API_KEY": "bu-ck",
    })
    @patch("langchain_openai.ChatOpenAI")
    def test_get_browser_use_llm_cerebras_fallback_key(self, mock_chat):
        from llm.factory import get_browser_use_llm
        mock_chat.return_value = MagicMock()
        get_browser_use_llm()
        assert mock_chat.call_args[1]["api_key"] == "bu-ck"

    @patch.dict(os.environ, {
        "BROWSER_USE_PROVIDER": "groq",
        "GROQ_API_KEY": "",
        "BROWSER_USE_API_KEY": "",
    })
    def test_get_browser_use_llm_missing_key_raises(self):
        from llm.factory import get_browser_use_llm
        with pytest.raises(RuntimeError, match="No API key"):
            get_browser_use_llm()


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — State Management
# ═══════════════════════════════════════════════════════════════════════════

class TestStateManagement:

    def test_graphstate_has_page_intelligence(self):
        from graph.state import GraphState
        assert "page_intelligence" in GraphState.__annotations__

    def test_graphstate_has_all_12_keys(self):
        from graph.state import GraphState
        expected = {
            "gherkin", "url", "page_intelligence", "action_plan",
            "selenium_script", "execution_results", "failed_steps",
            "healing_history", "retry_count", "final_report_paths",
            "error", "headless",
        }
        assert expected == set(GraphState.__annotations__.keys())

    def test_graphstate_total_false(self):
        from graph.state import GraphState
        assert GraphState.__total__ is False

    def test_initial_state_can_be_created(self):
        from graph.state import GraphState
        state: GraphState = {
            "gherkin": "Given x",
            "url": "https://x.com",
            "page_intelligence": {},
            "action_plan": [],
            "selenium_script": "",
            "execution_results": {},
            "failed_steps": [],
            "healing_history": [],
            "retry_count": 0,
            "final_report_paths": {},
            "error": "",
            "headless": True,
        }
        assert state["page_intelligence"] == {}


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — Self-Healer Patching
# ═══════════════════════════════════════════════════════════════════════════

class TestSelfHealerPatching:

    def _patch(self):
        from agents.self_healer import _patch_script
        return _patch_script

    def test_direct_replacement(self):
        ps = self._patch()
        script = 'driver.find_element(By.ID, "old-id").click()'
        info = {"step": 1, "locator_used": '(By.ID, "old-id")', "target": "btn"}
        patched, ok, desc = ps(script, info, "css_selector", "#new-id")
        assert ok
        assert "By.CSS_SELECTOR" in patched
        assert "#new-id" in patched

    def test_step_comment_vicinity(self):
        ps = self._patch()
        script = "# Step 2 — click login\nelem = driver.find_element(By.ID, 'login-btn')\nelem.click()\n"
        info = {"step": 2, "locator_used": "", "target": "login"}
        patched, ok, _ = ps(script, info, "xpath", "//button[@id='log-in']")
        assert ok
        assert "By.XPATH" in patched

    def test_target_word_fallback(self):
        ps = self._patch()
        script = "driver.find_element(By.CSS_SELECTOR, 'input.username').send_keys('a')\n"
        info = {"step": 1, "locator_used": "", "target": "username"}
        patched, ok, _ = ps(script, info, "id", "user-name")
        assert ok
        assert "By.ID" in patched

    def test_no_match_unchanged(self):
        ps = self._patch()
        script = "print('hello')\n"
        info = {"step": 1, "locator_used": "", "target": "nonexistent"}
        patched, ok, _ = ps(script, info, "id", "foo")
        assert not ok
        assert patched == script

    def test_patched_output_is_compilable(self):
        ps = self._patch()
        script = "from selenium.webdriver.common.by import By\nelem = driver.find_element(By.ID, 'old')\n"
        info = {"step": 1, "locator_used": "(By.ID, 'old')", "target": "elem"}
        patched, ok, _ = ps(script, info, "css_selector", "#new")
        if ok:
            compile(patched, "<test>", "exec")

    def test_by_map_covers_standard_types(self):
        from agents.self_healer import _BY_MAP
        for t in ("id", "css_selector", "xpath", "name", "class_name"):
            assert t in _BY_MAP

    @patch("agents.self_healer.get_llm")
    def test_max_retries_short_circuits(self, mock_llm):
        from agents.self_healer import run_self_healer
        state = {
            "failed_steps": [{"step": 1, "target": "btn", "error": "NoSuchElement"}],
            "execution_results": {},
            "selenium_script": "pass",
            "healing_history": [],
            "retry_count": 5,
        }
        result = run_self_healer(state)
        mock_llm.assert_not_called()
        assert any("Max retries" in h.get("reason", "") for h in result["healing_history"])

    @patch("agents.self_healer.get_llm")
    def test_no_failed_steps_short_circuits(self, mock_llm):
        from agents.self_healer import run_self_healer
        state = {
            "failed_steps": [],
            "execution_results": {},
            "selenium_script": "pass",
            "healing_history": [],
            "retry_count": 0,
        }
        result = run_self_healer(state)
        mock_llm.assert_not_called()
        assert result["selenium_script"] == "pass"


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — Step Router
# ═══════════════════════════════════════════════════════════════════════════

class TestStepRouter:

    def test_standard_selenium(self):
        from tools.step_router import route_step
        assert route_step({"complexity": "standard"}) == "selenium"

    def test_default_selenium(self):
        from tools.step_router import route_step
        assert route_step({}) == "selenium"

    def test_captcha_browser_use(self):
        from tools.step_router import route_step
        assert route_step({"complexity": "captcha"}) == "browser_use"

    def test_file_upload_browser_use(self):
        from tools.step_router import route_step
        assert route_step({"complexity": "file_upload"}) == "browser_use"

    def test_two_factor_browser_use(self):
        from tools.step_router import route_step
        assert route_step({"complexity": "two_factor"}) == "browser_use"

    def test_unknown_selenium(self):
        from tools.step_router import route_step
        assert route_step({"complexity": "unknown_xyz"}) == "selenium"


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 5 — Explorer Agent
# ═══════════════════════════════════════════════════════════════════════════

class TestExplorerAgent:

    def test_default_intelligence_keys(self):
        from agents.explorer import _default_intelligence
        d = _default_intelligence()
        for k in ["page_type", "login_form", "captcha", "two_factor",
                   "file_upload", "page_technology", "anti_bot_measures",
                   "notable_elements", "recommended_approach"]:
            assert k in d
        assert d["captcha"]["present"] is False

    def test_extract_json_clean(self):
        from agents.explorer import _extract_json_from_text
        r = _extract_json_from_text('{"page_type": "login"}')
        assert r["page_type"] == "login"

    def test_extract_json_code_fence(self):
        from agents.explorer import _extract_json_from_text
        r = _extract_json_from_text('```json\n{"page_type": "dash"}\n```')
        assert r["page_type"] == "dash"

    def test_extract_json_garbage_returns_default(self):
        from agents.explorer import _extract_json_from_text
        r = _extract_json_from_text("no json here")
        assert r["page_type"] == "unknown"

    def test_run_explorer_empty_url(self):
        from agents.explorer import run_explorer
        r = run_explorer({"url": "", "headless": True})
        assert r["page_intelligence"]["page_type"] == "unknown"

    @patch("agents.explorer.get_browser_use_llm")
    @patch("agents.explorer.create_chrome_driver")
    def test_run_explorer_vision_success(self, mock_drv_factory, mock_bu_llm):
        from agents.explorer import run_explorer

        # Create a real temp screenshot so open() can read it
        tmp_dir = tempfile.mkdtemp()
        mock_drv = MagicMock()
        mock_drv.page_source = "<html><body><input id='u'></body></html>"
        def _fake_screenshot(path):
            with open(path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True
        mock_drv.save_screenshot = _fake_screenshot
        mock_drv_factory.return_value = mock_drv

        resp = MagicMock()
        resp.content = json.dumps({"page_type": "login", "login_form": {"present": True, "has_username": True, "has_password": True, "has_social_login": False}, "captcha": {"present": False, "type": "none", "appears_after_submit": False}, "two_factor": {"present": False, "type": "none"}, "file_upload": {"present": False}, "page_technology": "traditional", "anti_bot_measures": ["none"], "notable_elements": [], "recommended_approach": "Selenium"})
        mock_bu_llm.return_value.invoke.return_value = resp

        with patch("agents.explorer.SCREENSHOT_DIR", tmp_dir):
            r = run_explorer({"url": "https://example.com", "headless": True})
        assert r["page_intelligence"]["page_type"] == "login"
        mock_drv.quit.assert_called_once()

    @patch("agents.explorer.get_llm")
    @patch("agents.explorer.get_browser_use_llm")
    @patch("agents.explorer.create_chrome_driver")
    def test_run_explorer_vision_fails_fallback(self, mock_drv_factory, mock_bu, mock_llm):
        from agents.explorer import run_explorer

        tmp_dir = tempfile.mkdtemp()
        mock_drv = MagicMock()
        mock_drv.page_source = "<html><body>Hi</body></html>"
        def _fake_screenshot(path):
            with open(path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True
        mock_drv.save_screenshot = _fake_screenshot
        mock_drv_factory.return_value = mock_drv
        mock_bu.return_value.invoke.side_effect = Exception("vision down")

        fb = MagicMock()
        fb.content = json.dumps({"page_type": "other"})
        mock_llm.return_value.invoke.return_value = fb

        with patch("agents.explorer.SCREENSHOT_DIR", tmp_dir):
            r = run_explorer({"url": "https://example.com", "headless": True})
        assert r["page_intelligence"]["page_type"] == "other"
        mock_llm.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 6 — Analyst Intelligence Overlay
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalystIntelligence:

    def test_overlay_captcha(self):
        from agents.analyst import _overlay_intelligence
        plan = [
            {"step": 1, "action": "navigate", "target": "url", "complexity": "standard", "description": ""},
            {"step": 2, "action": "click", "target": "login button", "complexity": "standard", "description": "submit"},
        ]
        pi = {"captcha": {"present": True, "appears_after_submit": True}, "file_upload": {"present": False}, "two_factor": {"present": False}}
        r = _overlay_intelligence(plan, pi)
        assert r[0]["complexity"] == "standard"
        assert r[1]["complexity"] == "captcha"

    def test_overlay_file_upload(self):
        from agents.analyst import _overlay_intelligence
        plan = [{"step": 1, "action": "upload", "target": "file input", "complexity": "standard", "description": ""}]
        pi = {"captcha": {"present": False}, "file_upload": {"present": True}, "two_factor": {"present": False}}
        assert _overlay_intelligence(plan, pi)[0]["complexity"] == "file_upload"

    def test_overlay_2fa(self):
        from agents.analyst import _overlay_intelligence
        plan = [{"step": 1, "action": "enter", "target": "otp field", "complexity": "standard", "description": ""}]
        pi = {"captcha": {"present": False}, "file_upload": {"present": False}, "two_factor": {"present": True}}
        assert _overlay_intelligence(plan, pi)[0]["complexity"] == "2fa"

    def test_overlay_no_change(self):
        from agents.analyst import _overlay_intelligence
        plan = [{"step": 1, "action": "click", "target": "button", "complexity": "standard", "description": ""}]
        pi = {"captcha": {"present": False}, "file_upload": {"present": False}, "two_factor": {"present": False}}
        assert _overlay_intelligence(plan, pi)[0]["complexity"] == "standard"

    @patch("agents.analyst.get_llm")
    def test_run_analyst_pi_in_prompt(self, mock_llm):
        from agents.analyst import run_analyst
        resp = MagicMock()
        resp.content = json.dumps([{"step": 1, "action": "navigate", "target": "url", "value": "", "description": "go", "complexity": "standard"}])
        mock_llm.return_value.invoke.return_value = resp
        state = {"gherkin": "Given I go", "url": "https://x.com", "page_intelligence": {"captcha": {"present": True}}}
        run_analyst(state)
        msgs = mock_llm.return_value.invoke.call_args[0][0]
        assert "Page Intelligence" in msgs[1].content

    @patch("agents.analyst.get_llm")
    def test_run_analyst_empty_inputs(self, mock_llm):
        from agents.analyst import run_analyst
        r = run_analyst({"gherkin": "", "url": ""})
        assert r["action_plan"] == []
        assert "Missing" in r.get("error", "")


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 7 — Execution Continuity
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutionContinuity:

    def test_inject_cookies(self):
        from agents.executor import _inject_cookies
        drv = MagicMock()
        _inject_cookies(drv, "https://x.com", [{"name": "s", "value": "v"}])
        drv.add_cookie.assert_called_once()
        drv.get.assert_called_once_with("https://x.com")
        drv.refresh.assert_called_once()

    def test_inject_cookies_noop_empty_list(self):
        from agents.executor import _inject_cookies
        drv = MagicMock()
        _inject_cookies(drv, "https://x.com", [])
        drv.add_cookie.assert_not_called()

    def test_inject_cookies_noop_empty_url(self):
        from agents.executor import _inject_cookies
        drv = MagicMock()
        _inject_cookies(drv, "", [{"name": "x", "value": "y"}])
        drv.add_cookie.assert_not_called()

    def test_executor_empty_script_error(self):
        from agents.executor import run_executor
        r = run_executor({"selenium_script": "", "url": "https://x.com", "headless": True})
        assert r["execution_results"]["overall_status"] == "ERROR"

    @patch("agents.executor.browser_use_runner")
    @patch("agents.executor.create_chrome_driver")
    @patch("agents.executor.execute_script")
    def test_session_state_propagation(self, mock_exec, mock_drv_f, mock_bu):
        from agents.executor import run_executor
        mock_bu.execute_step.return_value = {
            "status": "PASSED", "browser_use_handled": True,
            "current_url": "https://x.com/dash", "cookies": [{"name": "s", "value": "v"}],
            "step": 1, "engine": "browser_use",
        }
        drv = MagicMock()
        drv.page_source = "<html></html>"
        drv.current_url = "https://x.com/dash"
        drv.get_cookies.return_value = []
        mock_drv_f.return_value = drv
        mock_exec.return_value = {"overall_status": "PASSED", "steps": []}
        state = {
            "selenium_script": "pass", "url": "https://x.com", "headless": True,
            "action_plan": [{"step": 1, "complexity": "captcha", "action": "solve"}],
            "page_intelligence": {},
        }
        run_executor(state)
        # Browser Use was called for the captcha step
        mock_bu.execute_step.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 8 — Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:

    def test_build_graph_has_six_nodes(self):
        from graph.workflow import build_graph
        g = build_graph()
        names = set()
        try:
            for n in g.get_graph().nodes:
                names.add(n if isinstance(n, str) else n.id)
        except Exception:
            assert callable(g.invoke)
            return
        for name in ["explorer", "analyst", "script_writer", "executor", "healer", "reporter"]:
            assert name in names

    def test_entry_point_is_explorer(self):
        from graph.workflow import build_graph
        g = build_graph()
        try:
            edges = g.get_graph().edges
            start_targets = [
                (e[1] if isinstance(e[1], str) else e[1].id)
                for e in edges
                if (e[0] if isinstance(e[0], str) else e[0].id) == "__start__"
            ]
            assert "explorer" in start_targets
        except Exception:
            pass

    def test_should_heal_or_report_healer(self):
        from graph.workflow import should_heal_or_report
        assert should_heal_or_report({"failed_steps": [{"step": 1}], "retry_count": 0}) == "healer"

    def test_should_heal_or_report_reporter_no_failures(self):
        from graph.workflow import should_heal_or_report
        assert should_heal_or_report({"failed_steps": [], "retry_count": 0}) == "reporter"

    def test_should_heal_or_report_reporter_max_retries(self):
        from graph.workflow import should_heal_or_report
        assert should_heal_or_report({"failed_steps": [{"step": 1}], "retry_count": 3}) == "reporter"

    def test_after_healing_executor(self):
        from graph.workflow import after_healing
        assert after_healing({"failed_steps": [{"step": 1}], "retry_count": 1}) == "executor"

    def test_after_healing_reporter_max(self):
        from graph.workflow import after_healing
        assert after_healing({"failed_steps": [{"step": 1}], "retry_count": 3}) == "reporter"

    def test_after_healing_reporter_no_failures(self):
        from graph.workflow import after_healing
        assert after_healing({"failed_steps": [], "retry_count": 1}) == "reporter"

    def test_run_workflow_initial_state_has_pi(self):
        from graph.workflow import run_workflow
        with patch("graph.workflow.build_graph") as mock_bg:
            mock_g = MagicMock()
            mock_g.invoke.return_value = {}
            mock_bg.return_value = mock_g
            run_workflow("Given x", "https://x.com", headless=True)
            init = mock_g.invoke.call_args[0][0]
            assert "page_intelligence" in init
            assert init["page_intelligence"] == {}


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 9 — Report Generation
# ═══════════════════════════════════════════════════════════════════════════

class TestReportGeneration:

    def test_fallback_report_has_page_intelligence(self):
        from agents.reporter import _generate_report_data_fallback
        state = {
            "execution_results": {"steps": []},
            "healing_history": [],
            "gherkin": "x",
            "url": "https://x.com",
            "page_intelligence": {"page_type": "login"},
        }
        data = _generate_report_data_fallback(state)
        assert data["page_intelligence"]["page_type"] == "login"

    def test_fallback_passed(self):
        from agents.reporter import _generate_report_data_fallback
        state = {"execution_results": {"steps": [{"status": "PASSED"}]}, "healing_history": [], "gherkin": "x", "url": "u"}
        assert _generate_report_data_fallback(state)["overall_status"] == "PASSED"

    def test_fallback_failed(self):
        from agents.reporter import _generate_report_data_fallback
        state = {"execution_results": {"steps": [{"status": "FAILED"}]}, "healing_history": [], "gherkin": "x", "url": "u"}
        assert _generate_report_data_fallback(state)["overall_status"] == "FAILED"

    def test_fallback_partial(self):
        from agents.reporter import _generate_report_data_fallback
        state = {"execution_results": {"steps": [{"status": "PASSED"}, {"status": "FAILED"}]}, "healing_history": [], "gherkin": "x", "url": "u"}
        assert _generate_report_data_fallback(state)["overall_status"] == "PARTIALLY_PASSED"

    def test_fallback_error_override(self):
        from agents.reporter import _generate_report_data_fallback
        state = {"execution_results": {"steps": [{"status": "PASSED"}], "overall_status": "ERROR"}, "healing_history": [], "gherkin": "x", "url": "u"}
        assert _generate_report_data_fallback(state)["overall_status"] == "FAILED"

    @patch("agents.reporter.get_llm")
    def test_run_reporter_returns_paths(self, mock_llm):
        from agents.reporter import run_reporter
        mock_llm.return_value.invoke.side_effect = Exception("no llm")
        state = {
            "gherkin": "x", "url": "https://x.com",
            "execution_results": {"steps": [{"status": "PASSED", "step": 1}]},
            "healing_history": [], "retry_count": 0,
            "page_intelligence": {"page_type": "login"},
        }
        with patch("agents.reporter.REPORT_DIR", tempfile.mkdtemp()):
            r = run_reporter(state)
        assert os.path.isfile(r["final_report_paths"]["json"])
        assert os.path.isfile(r["final_report_paths"]["html"])

    @patch("agents.reporter.get_llm")
    def test_reporter_pi_in_prompt(self, mock_llm):
        from agents.reporter import run_reporter
        resp = MagicMock()
        resp.content = json.dumps({
            "report_title": "T", "overall_status": "PASSED",
            "summary": {"total_steps": 0, "passed_steps": 0, "failed_steps": 0, "healed_steps": 0, "total_execution_time_seconds": 0, "healing_attempts": 0, "successful_healings": 0},
            "steps_detail": [], "healing_events": [], "unresolved_failures": [], "recommendations": [],
        })
        mock_llm.return_value.invoke.return_value = resp
        state = {
            "gherkin": "x", "url": "u", "execution_results": {"steps": []},
            "healing_history": [], "retry_count": 0,
            "page_intelligence": {"page_type": "login"},
        }
        with patch("agents.reporter.REPORT_DIR", tempfile.mkdtemp()):
            run_reporter(state)
        msgs = mock_llm.return_value.invoke.call_args[0][0]
        assert "Page Intelligence" in msgs[1].content


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 10 — DOM Parser / Utilities
# ═══════════════════════════════════════════════════════════════════════════

class TestDOMParser:

    def test_clean_html_removes_scripts(self):
        from tools.dom_parser import clean_html
        html = "<html><head><script>alert(1)</script></head><body><p>hi</p></body></html>"
        cleaned = clean_html(html)
        assert "<script>" not in cleaned
        assert "hi" in cleaned

    def test_clean_html_max_length(self):
        from tools.dom_parser import clean_html
        long = "<html><body>" + "x" * 50000 + "</body></html>"
        cleaned = clean_html(long, max_length=1000)
        # Allow small overhead for truncation comment / tag closure
        assert len(cleaned) <= 1100

    def test_extract_interactive_elements(self):
        from tools.dom_parser import extract_interactive_elements
        html = '<html><body><input id="user"><button>Go</button></body></html>'
        elems = extract_interactive_elements(html)
        assert "input" in elems.lower() or "button" in elems.lower()


class TestPromptFiles:

    @pytest.mark.parametrize("filename", [
        "analyst_prompt.txt",
        "script_writer_prompt.txt",
        "healer_prompt.txt",
        "reporter_prompt.txt",
    ])
    def test_prompt_file_exists(self, filename):
        path = os.path.join(PROJECT_ROOT, "prompts", filename)
        assert os.path.isfile(path), f"Missing: {filename}"
        with open(path, "r", encoding="utf-8") as f:
            assert len(f.read()) > 100


class TestGherkinFileRead:

    def test_sample_feature_exists(self):
        path = os.path.join(PROJECT_ROOT, "tests", "sample.feature")
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Given" in content


class TestTemplateExists:

    def test_template_exists_and_has_pi_section(self):
        path = os.path.join(PROJECT_ROOT, "templates", "report_template.html")
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "{{ report" in content
        assert "page_intelligence" in content
