"""Self-Healer Agent — analyzes DOM and generates alternative locators.

When the Executor flags a step as failed due to a locator issue, this
agent uses the LLM to inspect the cleaned DOM and produce alternative
locator strategies, then patches the Selenium script to use them.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from tools.dom_parser import clean_html, extract_interactive_elements

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "healer_prompt.txt")

MAX_RETRIES = 3


def _load_system_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    patterns = [
        r"```json\s*\n?(.*?)```",
        r"```\s*\n?(.*?)```",
        r"\{.*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                return json.loads(candidate.strip())
            except (json.JSONDecodeError, IndexError):
                continue

    raise ValueError(f"Could not extract valid JSON from healer response:\n{text[:500]}")


def _patch_script(
    original_script: str,
    step_info: Dict[str, Any],
    new_locator_type: str,
    new_locator_value: str,
) -> str:
    """Attempt to patch the Selenium script by replacing the failed locator.

    This performs a best-effort replacement — if the exact old locator
    can be found in the script, it is swapped for the new one.  Otherwise
    the original script is returned unchanged.
    """
    patched = original_script

    # Map locator type to Selenium By constant
    by_map = {
        "id": "By.ID",
        "css_selector": "By.CSS_SELECTOR",
        "xpath": "By.XPATH",
        "name": "By.NAME",
        "class_name": "By.CLASS_NAME",
        "tag_name": "By.TAG_NAME",
        "link_text": "By.LINK_TEXT",
        "partial_link_text": "By.PARTIAL_LINK_TEXT",
    }

    new_by = by_map.get(new_locator_type, "By.CSS_SELECTOR")
    escaped_value = new_locator_value.replace("'", "\\'")

    # Try to find and replace the old locator in the script
    old_locator = step_info.get("locator_used", "")
    if old_locator and old_locator in original_script:
        patched = original_script.replace(old_locator, f"({new_by}, '{escaped_value}')", 1)
        logger.info("Patched script: replaced '%s' with (%s, '%s')", old_locator, new_by, new_locator_value)
    else:
        # Try to find patterns that look like locator tuples for this step
        step_num = step_info.get("step", "")
        target = step_info.get("target", "")

        # Look for common locator patterns near the step
        patterns_to_try = [
            # (By.CSS_SELECTOR, "...") or (By.XPATH, "...")
            r'\(By\.\w+,\s*["\'][^"\']*' + re.escape(target.split()[0] if target else "___") + r'[^"\']*["\']',
            # find_element(By.XXX, "...")
            r'find_element\(By\.\w+,\s*["\'][^"\']*["\']\)',
        ]

        # As a fallback, inject a comment-based fix instruction
        if f"# Step {step_num}" in patched or f"step {step_num}" in patched.lower():
            # Add a fallback locator near the step
            fallback_code = (
                f"\n    # HEALED: Alternative locator for step {step_num} ({target})\n"
                f"    # Using: {new_by}, '{new_locator_value}'\n"
            )
            patched = re.sub(
                rf"(# [Ss]tep\s*{step_num}\b.*?\n)",
                rf"\1{fallback_code}",
                patched,
                count=1,
            )

        # More aggressive: find the find_element call for this step and replace locator
        # Look for the section of code handling this step number
        step_pattern = rf"(step.*?{step_num}.*?find_element\w*\()([^)]+)(\))"
        match = re.search(step_pattern, patched, re.DOTALL | re.IGNORECASE)
        if match:
            patched = patched[:match.start(2)] + f"{new_by}, '{escaped_value}'" + patched[match.end(2):]
            logger.info("Patched script via regex for step %s", step_num)

    return patched


def run_self_healer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Self-Healer agent node for the LangGraph workflow.

    Reads ``failed_steps``, ``execution_results``, ``selenium_script``,
    and ``healing_history`` from state.  Produces updated ``selenium_script``
    and ``healing_history``.
    """
    failed_steps = state.get("failed_steps", [])
    execution_results = state.get("execution_results", {})
    original_script = state.get("selenium_script", "")
    healing_history = list(state.get("healing_history", []))
    retry_count = state.get("retry_count", 0)

    if not failed_steps:
        logger.info("Self-Healer: no failed steps to heal")
        return {
            "healing_history": healing_history,
            "selenium_script": original_script,
            "retry_count": retry_count,
        }

    if retry_count >= MAX_RETRIES:
        logger.warning("Self-Healer: max retries (%d) exceeded, giving up", MAX_RETRIES)
        for step in failed_steps:
            healing_history.append({
                "step": step.get("step"),
                "target": step.get("target"),
                "original_error": step.get("error"),
                "healed": False,
                "reason": "Max retries exceeded",
                "attempts": retry_count,
            })
        return {
            "healing_history": healing_history,
            "selenium_script": original_script,
            "failed_steps": [],  # Clear to prevent more healing loops
            "retry_count": retry_count,
        }

    logger.info("Self-Healer agent starting — healing %d failed steps (retry %d/%d)",
                len(failed_steps), retry_count + 1, MAX_RETRIES)

    # Get page HTML
    page_html = execution_results.get("page_html", "")
    if not page_html:
        logger.warning("No page HTML available for healing analysis")
        return {
            "healing_history": healing_history,
            "selenium_script": original_script,
            "retry_count": retry_count + 1,
        }

    # Clean the DOM for LLM context
    cleaned_dom = clean_html(page_html, max_length=25_000)
    interactive_elements = extract_interactive_elements(page_html)

    system_prompt = _load_system_prompt()
    patched_script = original_script
    any_healed = False

    for step_info in failed_steps:
        step_num = step_info.get("step", "?")
        target = step_info.get("target", "unknown")
        error = step_info.get("error", "")
        locator_used = step_info.get("locator_used", "")

        logger.info("Healing step %s: target='%s', error='%s'", step_num, target, error[:100])

        user_message = (
            f"## Failed Element\n"
            f"- Description: {target}\n"
            f"- Action: {step_info.get('action', 'unknown')}\n"
            f"- Original locator: {locator_used}\n"
            f"- Error: {error}\n\n"
            f"## Interactive Elements on Page\n"
            f"```\n{interactive_elements[:5000]}\n```\n\n"
            f"## Page DOM (simplified)\n"
            f"```html\n{cleaned_dom[:20000]}\n```\n\n"
            f"Analyze the DOM and provide alternative locators for the element described as '{target}'."
        )

        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=2048,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            response = llm.invoke(messages)
            healing_result = _extract_json(response.content)

            healed = healing_result.get("healed", False)
            recommended = healing_result.get("recommended")

            healing_event = {
                "step": step_num,
                "target": target,
                "original_error": error,
                "original_locator": locator_used,
                "healed": healed,
                "alternatives": healing_result.get("alternatives", []),
                "analysis": healing_result.get("analysis", ""),
                "attempts": retry_count + 1,
            }

            if healed and recommended:
                new_type = recommended.get("locator_type", "css_selector")
                new_value = recommended.get("locator_value", "")

                healing_event["new_locator_type"] = new_type
                healing_event["new_locator_value"] = new_value

                patched_script = _patch_script(patched_script, step_info, new_type, new_value)
                any_healed = True
                logger.info("Step %s healed: new locator %s = '%s'", step_num, new_type, new_value)
            else:
                logger.warning("Step %s could not be healed: %s",
                             step_num, healing_result.get("analysis", "Unknown"))

            healing_history.append(healing_event)

        except Exception as exc:
            logger.error("Self-healer LLM call failed for step %s: %s", step_num, exc)
            healing_history.append({
                "step": step_num,
                "target": target,
                "original_error": error,
                "healed": False,
                "reason": f"LLM error: {str(exc)}",
                "attempts": retry_count + 1,
            })

    return {
        "selenium_script": patched_script if any_healed else original_script,
        "healing_history": healing_history,
        "retry_count": retry_count + 1,
        "failed_steps": failed_steps if any_healed else [],  # Clear if nothing was healed
    }
