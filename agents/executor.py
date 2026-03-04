"""Executor Agent — runs the generated Selenium script against the real website.

Sets up Chrome, injects popup handlers, executes the test script, and
collects detailed per-step results including screenshots on failure.
Flags failed steps that may benefit from self-healing.

Maintains a ``session_state`` that tracks the current URL and cookies
across both Selenium and Browser Use engines so that chained steps can
share session context.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from tools.selenium_runner import create_chrome_driver, execute_script, SCREENSHOT_DIR, SCREENSHOT_SUCCESS_DIR, SCREENSHOT_FAILURE_DIR, page_intel_cache
from tools.popup_handler import handle_popups
from tools.dom_parser import get_page_html
from tools.step_router import route_step
import tools.browser_use_runner as browser_use_runner

logger = logging.getLogger(__name__)


# ── Cookie injection helper ─────────────────────────────────────────

def _inject_cookies(driver, url: str, cookies: list) -> None:
    """Navigate to *url* and inject *cookies* into the Selenium driver."""
    if not url or not cookies:
        return
    try:
        driver.get(url)
        for ck in cookies:
            try:
                sel_cookie: dict = {
                    "name": ck.get("name", ""),
                    "value": ck.get("value", ""),
                }
                if ck.get("domain"):
                    sel_cookie["domain"] = ck["domain"]
                if ck.get("path"):
                    sel_cookie["path"] = ck["path"]
                if ck.get("secure") is not None:
                    sel_cookie["secure"] = ck["secure"]
                if ck.get("httpOnly") is not None:
                    sel_cookie["httpOnly"] = ck["httpOnly"]
                if "expires" in ck:
                    sel_cookie["expiry"] = int(ck["expires"])
                driver.add_cookie(sel_cookie)
            except Exception as ce:
                logger.debug("Skip cookie %s: %s", ck.get("name"), ce)
        driver.refresh()
        time.sleep(1)
        logger.info(
            "Injected session state into Selenium: url=%s cookies=%d",
            url, len(cookies),
        )
    except Exception as exc:
        logger.warning("Failed to inject session state: %s", exc)


def run_executor(state: Dict[str, Any]) -> Dict[str, Any]:
    """Executor agent node for the LangGraph workflow.

    Reads ``selenium_script``, ``url``, and ``headless`` from state.
    Produces ``execution_results`` and ``failed_steps``.
    """
    script_code = state.get("selenium_script", "")
    url = state.get("url", "")
    headless = state.get("headless", True)

    if not script_code:
        logger.error("Executor received empty script")
        return {
            "execution_results": {"overall_status": "ERROR", "error": "No script to execute"},
            "failed_steps": [],
        }

    # ── Session state — persists across Browser Use / Selenium ────
    session_state: Dict[str, Any] = {
        "current_url": url,
        "cookies": [],
    }

    # ── Pre-flight: check if any steps need Browser Use ──────────
    action_plan = state.get("action_plan", [])
    page_intelligence = state.get("page_intelligence", {})
    browser_use_results: List[Dict[str, Any]] = []
    browser_use_step_nums: set = set()

    for step_action in action_plan:
        if route_step(step_action) == "browser_use":
            step_num = step_action.get("step", 0)
            complexity = step_action.get("complexity", "?")
            logger.info(
                "Step %d routed to Browser Use (complexity=%s)",
                step_num,
                complexity,
            )

            # Inject captcha_type from page intelligence cache so the
            # Browser Use runner can pick the right prompt strategy.
            if complexity == "captcha":
                current_url = session_state["current_url"]
                pi = page_intel_cache.get(current_url, {})
                captcha_type = (
                    pi.get("captcha_type")
                    or pi.get("captcha", {}).get("type", "none")
                    if isinstance(pi.get("captcha"), dict)
                    else pi.get("captcha_type", "none")
                )
                step_action["captcha_type"] = captcha_type or "none"
                logger.info(
                    "Step %d: injected captcha_type=%s from page intel",
                    step_num, step_action["captcha_type"],
                )

            bu_result = browser_use_runner.execute_step(step_action, session_state["current_url"])
            browser_use_results.append(bu_result)
            browser_use_step_nums.add(step_num)

            # Update session state from Browser Use result
            if bu_result.get("browser_use_handled") and bu_result.get("status") == "PASSED":
                if bu_result.get("current_url"):
                    session_state["current_url"] = bu_result["current_url"]
                if bu_result.get("cookies"):
                    session_state["cookies"] = bu_result["cookies"]

    logger.info("Executor agent starting — launching Chrome (headless=%s)", headless)

    driver = None
    try:
        driver = create_chrome_driver(headless=headless)

        # ── Inject session state into Selenium if Browser Use ran ──
        if session_state["cookies"]:
            _inject_cookies(driver, session_state["current_url"], session_state["cookies"])

        # Execute the script (Selenium handles standard steps)
        results = execute_script(script_code, driver, url)
        logger.info("Execution completed — overall status: %s", results.get("overall_status", "UNKNOWN"))

        # Update session state from Selenium
        try:
            session_state["current_url"] = driver.current_url
            session_state["cookies"] = driver.get_cookies()
        except Exception:
            pass

        # Identify failed steps that are candidates for self-healing
        # Capture per-step DOM so self-healer uses the RIGHT page context
        failed_steps: List[Dict[str, Any]] = []
        for step in results.get("steps", []):
            if step.get("status") == "FAILED":
                error_type = step.get("error_type", "")
                error_msg = step.get("error", "")
                is_locator_error = any(
                    kw in error_type or kw in error_msg
                    for kw in [
                        "NoSuchElement",
                        "TimeoutException",
                        "ElementNotInteractable",
                        "ElementClickIntercepted",
                        "StaleElementReference",
                        "not found",
                        "Could not find",
                        "unable to locate",
                    ]
                )
                if is_locator_error:
                    # Per-step DOM: prefer DOM captured at failure time,
                    # fall back to current page source
                    step_html = step.get("page_html_on_failure", "")
                    # Bug 2 fix: use failed_url captured at failure moment
                    step_url = step.get("failed_url") or step.get("url_on_failure", "")
                    if not step_html:
                        try:
                            step_html = driver.page_source
                            if not step_url:
                                step_url = driver.current_url
                        except Exception:
                            step_html = results.get("page_html", "")
                            if not step_url:
                                step_url = url

                    failed_steps.append({
                        "step": step.get("step"),
                        "action": step.get("action", ""),
                        "target": step.get("target", ""),
                        "error": error_msg,
                        "error_type": error_type,
                        "locator_used": step.get("locator_used", ""),
                        "screenshot": step.get("screenshot", ""),
                        "page_html_on_failure": step_html,
                        "url_on_failure": step_url,
                        "failed_url": step_url,
                    })

        # Get current page HTML for healing context (fallback)
        page_html = ""
        try:
            page_html = driver.page_source
        except Exception:
            page_html = results.get("page_html", "")
        results["page_html"] = page_html

        # ── Tag Selenium steps with engine & merge Browser Use results ──
        for s in results.get("steps", []):
            s.setdefault("engine", "selenium")

        if browser_use_results:
            all_steps = results.get("steps", [])
            for bu in browser_use_results:
                all_steps.append(bu)
            all_steps.sort(key=lambda s: s.get("step", 0))
            results["steps"] = all_steps
            statuses = [s.get("status") for s in all_steps]
            if all(st == "PASSED" for st in statuses):
                results["overall_status"] = "PASSED"
            elif any(st == "FAILED" for st in statuses):
                if results.get("overall_status") != "FAILED":
                    results["overall_status"] = "PARTIALLY_PASSED"

        if failed_steps:
            logger.info("Found %d failed steps eligible for self-healing", len(failed_steps))
        else:
            logger.info("No failed steps require healing")

        return {
            "execution_results": results,
            "failed_steps": failed_steps,
        }

    except Exception as exc:
        logger.error("Executor agent crashed: %s", exc, exc_info=True)
        page_html = ""
        if driver:
            try:
                page_html = driver.page_source
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                driver.save_screenshot(os.path.join(SCREENSHOT_DIR, f"executor_crash_{ts}.png"))
            except Exception:
                pass
        return {
            "execution_results": {
                "overall_status": "ERROR",
                "error": f"Executor crash: {str(exc)}",
                "steps": [],
                "page_html": page_html,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
            },
            "failed_steps": [],
        }
    finally:
        if driver:
            try:
                driver.quit()
                logger.info("Chrome driver closed")
            except Exception:
                pass
