"""Analyst Agent — parses Gherkin test cases into structured action plans.

This agent receives a raw Gherkin test and a URL, then uses the configured
LLM provider to produce a JSON array of UI actions that Selenium can execute.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from llm.factory import get_llm

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "analyst_prompt.txt")


def _load_system_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json(text: str) -> List[Dict[str, Any]]:
    """Extract JSON array from LLM response, handling markdown fences."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from code fences
    patterns = [
        r"```json\s*\n?(.*?)```",
        r"```\s*\n?(.*?)```",
        r"\[.*\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                return json.loads(candidate.strip())
            except (json.JSONDecodeError, IndexError):
                continue

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:500]}")


def run_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyst agent node for the LangGraph workflow.

    Reads ``gherkin`` and ``url`` from state, produces ``action_plan``.
    """
    gherkin: str = state.get("gherkin", "")
    url: str = state.get("url", "")

    if not gherkin or not url:
        logger.error("Analyst agent received empty gherkin or url")
        return {"action_plan": [], "error": "Missing gherkin or url input"}

    logger.info("Analyst agent starting — parsing Gherkin test case")
    logger.debug("Gherkin:\n%s", gherkin)
    logger.debug("URL: %s", url)

    system_prompt = _load_system_prompt()

    user_message = (
        f"Gherkin Test Case:\n```\n{gherkin}\n```\n\n"
        f"Target URL: {url}\n\n"
        "Produce the JSON action plan now."
    )

    try:
        llm = get_llm(temperature=0.1, max_tokens=4096)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        raw_response = response.content
        logger.debug("Analyst LLM raw response:\n%s", raw_response)

        action_plan = _extract_json(raw_response)
        logger.info("Analyst agent produced %d actions", len(action_plan))

        # Validate structure
        for i, action in enumerate(action_plan):
            if "step" not in action:
                action["step"] = i + 1
            if "action" not in action:
                raise ValueError(f"Action at index {i} missing 'action' key")
            if "target" not in action:
                raise ValueError(f"Action at index {i} missing 'target' key")
            action.setdefault("value", "")
            action.setdefault("description", "")

        return {"action_plan": action_plan}

    except Exception as exc:
        logger.error("Analyst agent failed: %s", exc, exc_info=True)
        return {"action_plan": [], "error": f"Analyst agent error: {str(exc)}"}
