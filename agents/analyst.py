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

    Reads ``gherkin``, ``url``, and ``page_intelligence`` from state,
    produces ``action_plan``.
    """
    gherkin: str = state.get("gherkin", "")
    url: str = state.get("url", "")
    page_intelligence: Dict[str, Any] = state.get("page_intelligence", {})

    if not gherkin or not url:
        logger.error("Analyst agent received empty gherkin or url")
        return {"action_plan": [], "error": "Missing gherkin or url input"}

    logger.info("Analyst agent starting — parsing Gherkin test case")
    logger.debug("Gherkin:\n%s", gherkin)
    logger.debug("URL: %s", url)

    system_prompt = _load_system_prompt()

    # Build page intelligence context for the LLM
    pi_context = ""
    if page_intelligence:
        # Detect captcha / 2FA presence for explicit instructions
        _pi_captcha = page_intelligence.get("captcha", {})
        if isinstance(_pi_captcha, bool):
            _pi_captcha = {"present": _pi_captcha}
        _captcha_detected = (
            _pi_captcha.get("present", False)
            or page_intelligence.get("captcha_present", False)
        )
        _captcha_type = (
            _pi_captcha.get("type", "none")
            or page_intelligence.get("captcha_type", "none")
        )
        _pi_tfa = page_intelligence.get("two_factor", {})
        if isinstance(_pi_tfa, bool):
            _pi_tfa = {"present": _pi_tfa}
        _tfa_detected = (
            _pi_tfa.get("present", False)
            or page_intelligence.get("two_factor_present", False)
        )

        captcha_instruction = ""
        if _captcha_detected or (_captcha_type and _captcha_type != "none"):
            captcha_instruction = (
                "\n### CAPTCHA DETECTED — MANDATORY RULE\n"
                f"The Explorer detected a captcha (type={_captcha_type}).\n"
                "You MUST generate exactly ONE step for the entire captcha:\n"
                '{"action": "solve_captcha", "target": "captcha", '
                '"complexity": "captcha", '
                '"description": "Solve the captcha on the page using Browser Use"}\n'
                "Do NOT break captcha into click + type steps. Browser Use handles it as one unit.\n\n"
            )

        tfa_instruction = ""
        if _tfa_detected:
            tfa_instruction = (
                "\n### 2FA DETECTED — MANDATORY RULE\n"
                "The Explorer detected two-factor authentication.\n"
                "You MUST generate exactly ONE step for the 2FA:\n"
                '{"action": "enter_otp", "target": "OTP input field", '
                '"complexity": "two_factor", '
                '"description": "Enter the 2FA/OTP verification code"}\n'
                "Do NOT break 2FA into locate + type + submit steps.\n\n"
            )

        pi_context = (
            "\n## Page Intelligence (gathered by Explorer agent)\n"
            f"```json\n{json.dumps(page_intelligence, indent=2, default=str)[:3000]}\n```\n\n"
            "Use this intelligence to tag step complexity correctly:\n"
            "- If captcha is present, tag solve_captcha steps as 'captcha'.\n"
            "- File uploads are ALWAYS handled by Selenium — tag upload steps as 'standard'.\n"
            "- If two_factor / 2FA is present, tag enter_otp/two_factor steps as 'two_factor'.\n"
            "- NEVER tag navigate/wait/assert/click/type steps as complex.\n"
            "- If anti-bot measures are detected, consider adding extra wait times.\n\n"
            f"{captcha_instruction}"
            f"{tfa_instruction}"
        )

    user_message = (
        f"Gherkin Test Case:\n```\n{gherkin}\n```\n\n"
        f"Target URL: {url}\n\n"
        f"{pi_context}"
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
            action.setdefault("complexity", "standard")

        # ── Overlay page-intelligence onto complexity tags ───────
        if page_intelligence:
            action_plan = _overlay_intelligence(action_plan, page_intelligence)

        return {"action_plan": action_plan}

    except Exception as exc:
        logger.error("Analyst agent failed: %s", exc, exc_info=True)
        return {"action_plan": [], "error": f"Analyst agent error: {str(exc)}"}


# ── Page intelligence → complexity overlay ──────────────────────────
# Only these EXACT action types trigger complexity overrides.
# All other actions (navigate, wait, assert, click, type, etc.) stay "standard".

def _overlay_intelligence(
    action_plan: List[Dict[str, Any]],
    pi: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Override complexity tags using Explorer-provided page intelligence.

    Rules (Fix 2):
    * action == "solve_captcha" AND captcha detected → "captcha"
    * action in ("enter_otp", "two_factor") AND 2FA detected → "two_factor"
    * action == "upload_file" → ALWAYS "standard" (Selenium handles it)
    * ALL other actions → keep "standard" regardless of page intel
    """
    captcha = pi.get("captcha", {})
    if isinstance(captcha, bool):
        captcha = {"present": captcha}
    captcha_present = (
        captcha.get("present", False)
        or pi.get("captcha_present", False)
    )

    two_factor = pi.get("two_factor", {})
    if isinstance(two_factor, bool):
        two_factor = {"present": two_factor}
    tfa_present = (
        two_factor.get("present", False)
        or pi.get("two_factor_present", False)
    )

    for step in action_plan:
        action_lower = step.get("action", "").lower().strip()

        # solve_captcha → captcha (if page actually has one)
        if action_lower == "solve_captcha" and captcha_present:
            step["complexity"] = "captcha"
            logger.debug("Step %s tagged as captcha by explorer intel", step.get("step"))
            continue

        # enter_otp / two_factor → two_factor (if page actually has 2FA)
        if action_lower in ("enter_otp", "two_factor") and tfa_present:
            step["complexity"] = "two_factor"
            logger.debug("Step %s tagged as two_factor by explorer intel", step.get("step"))
            continue

        # upload_file → always standard (Selenium handles uploads)
        if action_lower == "upload_file":
            step["complexity"] = "standard"
            logger.debug("Step %s forced to standard (file upload via Selenium)", step.get("step"))
            continue

        # Everything else: force standard — never tag click/navigate/type as complex
        step["complexity"] = "standard"

    return action_plan