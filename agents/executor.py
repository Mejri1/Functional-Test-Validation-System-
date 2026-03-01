"""Executor Agent — runs the generated Selenium script against the real website.

Sets up Chrome, injects popup handlers, executes the test script, and
collects detailed per-step results including screenshots on failure.
Flags failed steps that may benefit from self-healing.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from tools.selenium_runner import create_chrome_driver, execute_script, SCREENSHOT_DIR, SCREENSHOT_SUCCESS_DIR, SCREENSHOT_FAILURE_DIR
from tools.popup_handler import handle_popups
from tools.dom_parser import get_page_html

logger = logging.getLogger(__name__)


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

    logger.info("Executor agent starting — launching Chrome (headless=%s)", headless)

    driver = None
    try:
        driver = create_chrome_driver(headless=headless)

        # Execute the script
        results = execute_script(script_code, driver, url)
        logger.info("Execution completed — overall status: %s", results.get("overall_status", "UNKNOWN"))

        # Identify failed steps that are candidates for self-healing
        failed_steps: List[Dict[str, Any]] = []
        for step in results.get("steps", []):
            if step.get("status") == "FAILED":
                error_type = step.get("error_type", "")
                error_msg = step.get("error", "")
                # Only heal locator-related errors
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
                    failed_steps.append({
                        "step": step.get("step"),
                        "action": step.get("action", ""),
                        "target": step.get("target", ""),
                        "error": error_msg,
                        "error_type": error_type,
                        "locator_used": step.get("locator_used", ""),
                        "screenshot": step.get("screenshot", ""),
                    })

        # Get current page HTML for healing context
        page_html = ""
        try:
            page_html = driver.page_source
        except Exception:
            page_html = results.get("page_html", "")

        results["page_html"] = page_html

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
