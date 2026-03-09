"""Explorer Agent — visits the target URL and gathers page intelligence.

Runs **before** the Analyst.  Uses Selenium to capture a screenshot and
page HTML, then sends them to the vision-capable Browser Use LLM to
produce a structured JSON describing the page (login form, captcha,
2FA, file-upload, anti-bot measures, etc.).

The resulting ``page_intelligence`` dict is stored in state and consumed
by the Analyst for smarter complexity tagging.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict
from selenium.webdriver.support.ui import WebDriverWait

from langchain_core.messages import HumanMessage

from llm.factory import get_browser_use_llm, get_llm
from tools.selenium_runner import create_chrome_driver
from tools.dom_parser import clean_html

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_DIR = os.path.join(BASE_DIR, "outputs", "screenshots", "explorer")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ── Vision prompt ────────────────────────────────────────────────────
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "explorer_prompt.txt")
def _load_system_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

_EXPLORER_PROMPT = _load_system_prompt()

def wait_for_page_load(driver, timeout=20):
    # Step 1: DOM ready
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    # Step 2: For SPAs, wait for JS frameworks to finish rendering
    import time
    time.sleep(0.8)  # small intentional pause — SPAs need this, not negotiable
def _default_intelligence() -> Dict[str, Any]:
    """Return a safe-default page_intelligence dict."""
    return {
        "page_type": "unknown",
        "login_form": {"present": False, "has_username": False, "has_password": False, "has_social_login": False},
        "captcha": {"present": False, "type": "none", "appears_after_submit": False},
        "two_factor": {"present": False, "type": "none"},
        "file_upload": {"present": False},
        "page_technology": "unknown",
        "anti_bot_measures": ["none"],
        "notable_elements": [],
        "recommended_approach": "",
    }


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM response."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    patterns = [
        r"```json\s*\n?(.*?)```",
        r"```\s*\n?(.*?)```",
        r"\{[\s\S]*\}",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                candidate = m.group(1) if m.lastindex else m.group(0)
                return json.loads(candidate.strip())
            except (json.JSONDecodeError, IndexError):
                continue
    return _default_intelligence()


def run_explorer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Explorer agent node — gathers page intelligence for the Analyst.

    Reads ``url`` and ``headless`` from state.
    Produces ``page_intelligence``.
    """
    url = state.get("url", "")
    headless = state.get("headless", True)

    if not url:
        logger.error("Explorer received empty URL")
        return {"page_intelligence": _default_intelligence()}

    logger.info("Explorer agent starting — visiting %s", url)

    driver = None
    screenshot_b64 = ""
    page_html = ""

    try:
        driver = create_chrome_driver(headless=headless)
        driver.get(url)

        # Wait for full page load
        wait_for_page_load(driver, timeout=20)

        # Capture screenshot
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ss_path = os.path.join(SCREENSHOT_DIR, f"explorer_{ts}.png")
        driver.save_screenshot(ss_path)
        with open(ss_path, "rb") as f:
            screenshot_b64 = base64.b64encode(f.read()).decode("ascii")

        # Capture DOM
        page_html = driver.page_source
        cleaned_html = clean_html(page_html, max_length=15_000)

        logger.info("Explorer: captured screenshot and %d chars of HTML", len(page_html))

    except Exception as exc:
        logger.error("Explorer page visit failed: %s", exc)
        return {"page_intelligence": _default_intelligence()}
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    # ── Send to vision LLM ──────────────────────────────────────────
    try:
        vision_llm = get_browser_use_llm()

        content_parts = [
            {"type": "text", "text": f"{_EXPLORER_PROMPT}\n\nPage HTML (simplified):\n```html\n{cleaned_html}\n```"},
        ]
        if screenshot_b64:
            content_parts.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            })

        msg = HumanMessage(content=content_parts)
        response = vision_llm.invoke([msg])
        intelligence = _extract_json_from_text(response.content)

    except Exception as exc:
        logger.warning("Explorer vision LLM failed (%s), falling back to HTML-only analysis", exc)
        # Try text-only analysis with the regular LLM
        try:
            fallback_llm = get_llm(temperature=0.1, max_tokens=2048)
            msg = HumanMessage(content=f"{_EXPLORER_PROMPT}\n\nPage HTML:\n```html\n{cleaned_html[:12000]}\n```")
            response = fallback_llm.invoke([msg])
            intelligence = _extract_json_from_text(response.content)
        except Exception as exc2:
            logger.error("Explorer fallback LLM also failed: %s", exc2)
            intelligence = _default_intelligence()

    # Ensure all expected keys exist
    default = _default_intelligence()
    for key in default:
        intelligence.setdefault(key, default[key])

    # ── Log summary ─────────────────────────────────────────────────
    login = "yes" if intelligence.get("login_form", {}).get("present") else "no"
    captcha_type = intelligence.get("captcha", {}).get("type", "none")
    tfa = "yes" if intelligence.get("two_factor", {}).get("present") else "no"
    page_type = intelligence.get("page_type", "unknown")

    logger.info(
        "Explorer: page_type=%s, login_form=%s, captcha=%s, 2fa=%s",
        page_type, login, captcha_type, tfa,
    )
    print(f"   Explorer: page_type={page_type}, login_form={login}, "
          f"captcha={captcha_type}, 2fa={tfa}")

    return {"page_intelligence": intelligence}
