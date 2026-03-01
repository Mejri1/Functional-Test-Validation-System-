"""Script Writer Agent — generates a runnable Selenium Python script.

Takes the structured action plan from the Analyst and produces a complete
Python script with robust locator strategies, error handling, and screenshot
capture.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "script_writer_prompt.txt")


def _load_system_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _extract_python_code(text: str) -> str:
    """Extract Python code from LLM response."""
    text = text.strip()

    # Try to extract from code fences
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```py\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no code fences, check if it looks like Python code
    if "import " in text or "def " in text:
        # Remove any leading non-code text
        lines = text.split("\n")
        code_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ", "#!", "# ", "def ", '"""')):
                code_start = i
                break
        return "\n".join(lines[code_start:])

    return text


def run_script_writer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Script Writer agent node for the LangGraph workflow.

    Reads ``action_plan`` and ``url`` from state, produces ``selenium_script``.
    """
    action_plan = state.get("action_plan", [])
    url = state.get("url", "")

    if not action_plan:
        logger.error("Script Writer received empty action plan")
        return {"selenium_script": "", "error": "No action plan provided"}

    logger.info("Script Writer agent starting — generating Selenium script for %d actions", len(action_plan))

    system_prompt = _load_system_prompt()

    user_message = (
        f"Action Plan:\n```json\n{json.dumps(action_plan, indent=2)}\n```\n\n"
        f"Target URL: {url}\n\n"
        "Generate the complete Selenium Python script now. "
        "Remember: the script must define a `run_test(driver)` function that accepts "
        "a Selenium WebDriver and returns a results dictionary."
    )

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=8192,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        raw_response = response.content
        logger.debug("Script Writer LLM raw response length: %d chars", len(raw_response))

        script_code = _extract_python_code(raw_response)

        # Validate the script has essential components
        if "def run_test" not in script_code:
            logger.warning("Generated script missing run_test function, adding wrapper")
            script_code = _wrap_in_run_test(script_code)

        if "from selenium" not in script_code and "import selenium" not in script_code:
            logger.warning("Generated script missing selenium imports, adding them")
            script_code = _add_selenium_imports() + "\n\n" + script_code

        logger.info("Script Writer agent produced script (%d characters)", len(script_code))
        return {"selenium_script": script_code}

    except Exception as exc:
        logger.error("Script Writer agent failed: %s", exc, exc_info=True)
        return {"selenium_script": "", "error": f"Script Writer error: {str(exc)}"}


def _add_selenium_imports() -> str:
    return """import os
import json
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException,
    ElementClickInterceptedException, StaleElementReferenceException
)"""


def _wrap_in_run_test(code: str) -> str:
    """Wrap loose code into a run_test function."""
    imports = _add_selenium_imports()
    return f"""{imports}

SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def run_test(driver):
    results = {{
        "steps": [],
        "overall_status": "PASSED",
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }}

    try:
{_indent(code, 8)}
    except Exception as e:
        results["overall_status"] = "ERROR"
        results["error"] = str(e)

    results["end_time"] = datetime.now().isoformat()
    return results
"""


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.split("\n"))
