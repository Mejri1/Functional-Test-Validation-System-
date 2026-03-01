"""Sample test to validate the system components work end-to-end.

Run with:  python -m pytest tests/test_sample.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Unit tests for individual components ────────────────────────────


class TestDomParser:
    """Tests for the DOM parser utility."""

    def test_clean_html_removes_scripts(self):
        from tools.dom_parser import clean_html

        html = "<html><body><script>alert('x')</script><div id='main'>Hello</div></body></html>"
        cleaned = clean_html(html)
        assert "alert" not in cleaned
        assert "Hello" in cleaned

    def test_clean_html_keeps_ids_and_classes(self):
        from tools.dom_parser import clean_html

        html = '<html><body><div id="login" class="form-group"><input name="email" type="email" placeholder="Email"></div></body></html>'
        cleaned = clean_html(html)
        assert 'id="login"' in cleaned
        assert 'name="email"' in cleaned
        assert 'type="email"' in cleaned

    def test_extract_interactive_elements(self):
        from tools.dom_parser import extract_interactive_elements

        html = """
        <html><body>
            <form><input type="email" name="email" placeholder="Enter email">
            <button type="submit">Login</button></form>
            <div>Non-interactive</div>
        </body></html>
        """
        result = extract_interactive_elements(html)
        assert "input" in result
        assert "button" in result
        assert "Login" in result

    def test_clean_html_truncation(self):
        from tools.dom_parser import clean_html

        html = "<html><body>" + "<p>x</p>" * 10000 + "</body></html>"
        cleaned = clean_html(html, max_length=500)
        assert len(cleaned) <= 600  # Allow some overhead for truncation message


class TestPopupHandler:
    """Tests for the popup handler (JS generation only, no browser needed)."""

    def test_js_dismisser_is_valid_string(self):
        from tools.popup_handler import POPUP_DISMISSER_JS

        assert isinstance(POPUP_DISMISSER_JS, str)
        assert "cookie" in POPUP_DISMISSER_JS.lower()
        assert "MutationObserver" in POPUP_DISMISSER_JS


class TestGraphState:
    """Tests for the state definition."""

    def test_state_keys_exist(self):
        from graph.state import GraphState

        # TypedDict creates a class with __annotations__
        expected_keys = {
            "gherkin", "url", "action_plan", "selenium_script",
            "execution_results", "failed_steps", "healing_history",
            "retry_count", "final_report_paths", "error", "headless",
        }
        actual_keys = set(GraphState.__annotations__.keys())
        assert expected_keys == actual_keys


class TestAnalystJsonExtraction:
    """Test the JSON extraction helper in the analyst agent."""

    def test_extract_json_from_raw(self):
        from agents.analyst import _extract_json

        raw = '[{"step": 1, "action": "navigate", "target": "url", "value": "https://example.com"}]'
        result = _extract_json(raw)
        assert len(result) == 1
        assert result[0]["action"] == "navigate"

    def test_extract_json_from_markdown(self):
        from agents.analyst import _extract_json

        raw = '```json\n[{"step": 1, "action": "click", "target": "button"}]\n```'
        result = _extract_json(raw)
        assert result[0]["action"] == "click"


class TestScriptWriterCodeExtraction:
    """Test the Python code extraction in the script writer."""

    def test_extract_from_fenced_code(self):
        from agents.script_writer import _extract_python_code

        raw = "Here is the script:\n```python\nimport os\ndef run_test(driver):\n    pass\n```\nDone."
        code = _extract_python_code(raw)
        assert "def run_test(driver)" in code
        assert "import os" in code

    def test_extract_from_raw_code(self):
        from agents.script_writer import _extract_python_code

        raw = "import os\nimport json\ndef run_test(driver):\n    return {}"
        code = _extract_python_code(raw)
        assert "def run_test" in code


class TestHealerJsonExtraction:
    """Test JSON extraction in the self-healer."""

    def test_extract_healing_result(self):
        from agents.self_healer import _extract_json

        raw = json.dumps({
            "healed": True,
            "recommended": {"locator_type": "css_selector", "locator_value": "#btn"},
            "alternatives": [],
            "analysis": "Found button",
        })
        result = _extract_json(raw)
        assert result["healed"] is True
        assert result["recommended"]["locator_value"] == "#btn"


class TestWorkflowBuilds:
    """Test that the LangGraph workflow compiles without error."""

    def test_graph_compiles(self):
        from graph.workflow import build_graph

        graph = build_graph()
        assert graph is not None


class TestGherkinFileRead:
    """Test that the sample feature file is readable."""

    def test_sample_feature_exists(self):
        feature_path = os.path.join(ROOT, "tests", "sample.feature")
        assert os.path.isfile(feature_path)
        with open(feature_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Given" in content
        assert "Then" in content


class TestPromptFiles:
    """Verify all prompt files exist and are non-empty."""

    @pytest.mark.parametrize("filename", [
        "analyst_prompt.txt",
        "script_writer_prompt.txt",
        "healer_prompt.txt",
        "reporter_prompt.txt",
    ])
    def test_prompt_file_exists(self, filename):
        path = os.path.join(ROOT, "prompts", filename)
        assert os.path.isfile(path), f"Missing prompt file: {filename}"
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100, f"Prompt file {filename} seems too short"


class TestTemplateExists:
    """Verify the HTML report template exists."""

    def test_template_exists(self):
        path = os.path.join(ROOT, "templates", "report_template.html")
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "{{ report" in content
