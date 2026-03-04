"""Self-Healer Agent — analyzes DOM and generates alternative locators.

When the Executor flags a step as failed due to a locator issue, this
agent uses the LLM to inspect the cleaned DOM *captured at the time of
failure* (per-step, not a single shared DOM) and produce alternative
locator strategies.  It then **patches the Selenium script** to use
the new locators, validates the patch with ``compile()``, and saves
the healed script.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from llm.factory import get_llm

from tools.dom_parser import clean_html, extract_interactive_elements

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "healer_prompt.txt")
SCRIPT_DIR = os.path.join(BASE_DIR, "outputs", "generated_scripts")
os.makedirs(SCRIPT_DIR, exist_ok=True)

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


# ── Locator type → Selenium By constant ─────────────────────────────

_BY_MAP = {
    "id": "By.ID",
    "css_selector": "By.CSS_SELECTOR",
    "xpath": "By.XPATH",
    "name": "By.NAME",
    "class_name": "By.CLASS_NAME",
    "tag_name": "By.TAG_NAME",
    "link_text": "By.LINK_TEXT",
    "partial_link_text": "By.PARTIAL_LINK_TEXT",
}


def _patch_script(
    script: str,
    step_info: Dict[str, Any],
    new_locator_type: str,
    new_locator_value: str,
) -> tuple[str, bool, str]:
    """Attempt to patch the Selenium script by replacing a failed locator.

    Returns ``(patched_script, was_patched, description)``.
    """
    new_by = _BY_MAP.get(new_locator_type, "By.CSS_SELECTOR")
    escaped_value = new_locator_value.replace("\\", "\\\\").replace("'", "\\'")
    new_locator_str = f"{new_by}, '{escaped_value}'"

    old_locator = step_info.get("locator_used", "")
    step_num = step_info.get("step", "?")

    # Strategy 1: direct string replacement of the old locator tuple
    if old_locator and old_locator in script:
        patched = script.replace(old_locator, f"({new_locator_str})", 1)
        desc = f"Step {step_num}: replaced '{old_locator}' with '({new_locator_str})'"
        return patched, True, desc

    # Strategy 2: find_element/find_elements call in the step's vicinity
    # Look for patterns like find_element(By.XXX, "...") near step comment
    step_pattern = re.compile(
        rf"(#.*?[Ss]tep\s*{step_num}\b[^\n]*\n"
        rf"(?:[^\n]*\n){{0,8}}?"     # up to 8 lines after step comment
        rf")(find_elements?\()(By\.\w+,\s*['\"][^'\"]*['\"])"
        rf"(\))",
        re.DOTALL,
    )
    m = step_pattern.search(script)
    if m:
        patched = script[:m.start(3)] + new_locator_str + script[m.end(3):]
        desc = f"Step {step_num}: replaced '{m.group(3)}' with '{new_locator_str}'"
        return patched, True, desc

    # Strategy 3: any find_element with old target text
    target = step_info.get("target", "")
    if target:
        # Try to find a find_element call containing part of the target
        target_word = re.escape(target.split()[0]) if target.split() else ""
        if target_word:
            fe_pattern = re.compile(
                rf"(find_elements?\()(By\.\w+,\s*['\"][^'\"]*{target_word}[^'\"]*['\"])(\))",
                re.IGNORECASE,
            )
            m2 = fe_pattern.search(script)
            if m2:
                patched = script[:m2.start(2)] + new_locator_str + script[m2.end(2):]
                desc = f"Step {step_num}: replaced '{m2.group(2)}' with '{new_locator_str}'"
                return patched, True, desc

    logger.warning(
        "Step %s: could not find locator in script to replace (old_locator='%s', target='%s')",
        step_num, old_locator, target,
    )
    return script, False, f"Step {step_num}: locator not found in script — skipped"


def run_self_healer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Self-Healer agent node for the LangGraph workflow.

    Uses **per-step DOM** from ``step["page_html_on_failure"]`` instead of
    a single shared DOM.  After healing, patches the Selenium script,
    validates with ``compile()``, and saves to disk.
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
            "failed_steps": [],
            "retry_count": retry_count,
        }

    logger.info("Self-Healer agent starting — healing %d failed steps (retry %d/%d)",
                len(failed_steps), retry_count + 1, MAX_RETRIES)

    system_prompt = _load_system_prompt()
    patched_script = original_script
    any_healed = False

    for step_info in failed_steps:
        step_num = step_info.get("step", "?")
        target = step_info.get("target", "unknown")
        error = step_info.get("error", "")
        locator_used = step_info.get("locator_used", "")

        # ── Per-step DOM: use the DOM captured at failure time ────
        step_html = step_info.get("page_html_on_failure", "")
        # Bug 2 fix: prefer failed_url (captured at the exact moment of
        # failure) over url_on_failure (which may be the executor fallback)
        step_url = (
            step_info.get("failed_url")
            or step_info.get("url_on_failure", "")
        )

        # Fallback to shared DOM if per-step DOM not available
        if not step_html:
            step_html = execution_results.get("page_html", "")
            if not step_url:
                step_url = state.get("url", "")

        if not step_html:
            logger.warning("Step %s: no page HTML available for healing", step_num)
            healing_history.append({
                "step": step_num,
                "target": target,
                "original_error": error,
                "healed": False,
                "reason": "No page HTML available",
                "attempts": retry_count + 1,
            })
            continue

        logger.info("Healing step %s: target='%s', url='%s', error='%s'",
                     step_num, target, step_url, error[:100])

        cleaned_dom = clean_html(step_html, max_length=25_000)
        interactive_elements = extract_interactive_elements(step_html)

        user_message = (
            f"## Failed Element\n"
            f"- Description: {target}\n"
            f"- Action: {step_info.get('action', 'unknown')}\n"
            f"- Original locator: {locator_used}\n"
            f"- Error: {error}\n"
            f"- Page URL at failure: {step_url}\n\n"
            f"## Interactive Elements on Page\n"
            f"```\n{interactive_elements[:5000]}\n```\n\n"
            f"## Page DOM (simplified)\n"
            f"```html\n{cleaned_dom[:20000]}\n```\n\n"
            f"Analyze the DOM and provide alternative locators for the element described as '{target}'."
        )

        try:
            llm = get_llm(temperature=0.1, max_tokens=2048)

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
                "healed": False,
                "alternatives": healing_result.get("alternatives", []),
                "analysis": healing_result.get("analysis", ""),
                "attempts": retry_count + 1,
                "url_on_failure": step_url,
            }

            if healed and recommended:
                new_type = recommended.get("locator_type", "css_selector")
                new_value = recommended.get("locator_value", "")

                # Patch the script
                candidate_script, was_patched, desc = _patch_script(
                    patched_script, step_info, new_type, new_value,
                )

                if was_patched:
                    # Compile-check the patched script
                    try:
                        compile(candidate_script, "<string>", "exec")
                    except SyntaxError as se:
                        logger.error("Patched script syntax error: %s — reverting step %s", se, step_num)
                        healing_event["healed"] = False
                        healing_event["reason"] = f"Patch caused SyntaxError: {se}"
                    else:
                        patched_script = candidate_script
                        any_healed = True
                        healing_event["healed"] = True
                        healing_event["new_locator_type"] = new_type
                        healing_event["new_locator_value"] = new_value
                        logger.info("%s", desc)
                else:
                    logger.warning("Step %s: locator not found in script, skipping patch", step_num)
                    healing_event["reason"] = "Locator not found in script"
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

    # ── Save patched script if anything was healed ───────────────
    if any_healed:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        healed_path = os.path.join(SCRIPT_DIR, f"healed_script_{ts}.py")
        try:
            with open(healed_path, "w", encoding="utf-8") as f:
                f.write(patched_script)
            logger.info("Healed script saved to %s", healed_path)
        except Exception as exc:
            logger.error("Failed to save healed script: %s", exc)

    return {
        "selenium_script": patched_script if any_healed else original_script,
        "healing_history": healing_history,
        "retry_count": retry_count + 1,
        "failed_steps": failed_steps if any_healed else [],
    }
