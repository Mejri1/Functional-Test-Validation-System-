"""Selenium runner utility.

Sets up a Chrome WebDriver with the required options and provides
helpers for executing generated test scripts.
"""

from __future__ import annotations

import logging
import os
import tempfile
import importlib.util
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from tools.popup_handler import (
    handle_popups,
    handle_popups_light,
    inject_persistent_dismisser,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Driver wrapper — lightweight popup hooks after navigate / click / submit
# ---------------------------------------------------------------------------

def _install_popup_hooks(driver: webdriver.Chrome) -> webdriver.Chrome:
    """Monkey-patch *driver* so that a **lightweight** popup check runs
    after every action that may trigger a new page or overlay:

    * ``driver.get(url)`` — ensure JS dismisser + alert check (~1–3 ms)
    * ``element.click()`` — alert check only (~0–1 ms)
    * ``element.submit()`` — alert check only (~0–1 ms)

    The heavy JS observer (installed via CDP at driver creation) handles
    actual popup dismissal in the background.  These hooks only ensure
    the observer is alive and dismiss native browser alerts.
    """
    from selenium.webdriver.remote.webelement import WebElement

    # ---- driver.get() ----
    _original_get = driver.get

    def _get_with_popups(url: str) -> None:  # type: ignore[override]
        _original_get(url)
        try:
            handle_popups_light(driver)
        except Exception as exc:
            logger.debug("handle_popups_light after get() failed: %s", exc)

    driver.get = _get_with_popups  # type: ignore[assignment]

    # ---- element.click() / element.submit() ----
    _original_find_element = driver.find_element
    _original_find_elements = driver.find_elements

    def _wrap_element(el: WebElement) -> WebElement:
        """Patch click() and submit() on a single WebElement."""
        if getattr(el, "_popup_hooked", False):
            return el

        _orig_click = el.click
        _orig_submit = el.submit

        def _click_with_popups() -> None:
            _orig_click()
            try:
                handle_popups_light(driver)
            except Exception as exc:
                logger.debug("handle_popups_light after click() failed: %s", exc)

        def _submit_with_popups() -> None:
            _orig_submit()
            try:
                handle_popups_light(driver)
            except Exception as exc:
                logger.debug("handle_popups_light after submit() failed: %s", exc)

        el.click = _click_with_popups     # type: ignore[assignment]
        el.submit = _submit_with_popups   # type: ignore[assignment]
        el._popup_hooked = True            # type: ignore[attr-defined]
        return el

    def _find_element_wrapped(*args, **kwargs) -> WebElement:
        el = _original_find_element(*args, **kwargs)
        return _wrap_element(el)

    def _find_elements_wrapped(*args, **kwargs):
        elements = _original_find_elements(*args, **kwargs)
        return [_wrap_element(e) for e in elements]

    driver.find_element = _find_element_wrapped    # type: ignore[assignment]
    driver.find_elements = _find_elements_wrapped  # type: ignore[assignment]

    logger.info("Popup hooks installed on driver (get / click / submit)")
    return driver

# ── WebDriver Manager caching ───────────────────────────────────────
# Skip online version check if driver is already cached; re-check every N days.
os.environ.setdefault("WDM_CHECK_DRIVER_VERSION", os.getenv("WDM_CHECK_DRIVER_VERSION", "false"))
os.environ.setdefault("WDM_CACHE_VALID_RANGE", os.getenv("WDM_CACHE_VALID_RANGE", "7"))

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_DIR = os.path.join(BASE_DIR, "outputs", "screenshots")
SCREENSHOT_SUCCESS_DIR = os.path.join(SCREENSHOT_DIR, "success")
SCREENSHOT_FAILURE_DIR = os.path.join(SCREENSHOT_DIR, "failure")
SCRIPT_DIR = os.path.join(BASE_DIR, "outputs", "generated_scripts")
os.makedirs(SCREENSHOT_SUCCESS_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_FAILURE_DIR, exist_ok=True)
os.makedirs(SCRIPT_DIR, exist_ok=True)


def create_chrome_driver(headless: bool = True) -> webdriver.Chrome:
    """Create and return a configured Chrome WebDriver instance.

    Parameters
    ----------
    headless : bool
        Whether to run Chrome in headless mode (default True).
    """
    options = Options()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    # Suppress automation flags
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-blink-features=AutomationControlled")

    # Disable Chrome password manager / credential popups
    options.add_argument("--disable-save-password-bubble")
    options.add_argument("--disable-features=PasswordLeakDetection")
    options.add_argument("--password-store=basic")
    options.add_argument("--use-mock-keychain")

    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "profile.default_content_setting_values.notifications": 2,
    }
    options.add_experimental_option("prefs", prefs)

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception:
        logger.warning("webdriver-manager failed, falling back to system chromedriver")
        driver = webdriver.Chrome(options=options)

    # Stealth tweaks
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(5)

    logger.info("Chrome WebDriver created (headless=%s)", headless)

    # Install persistent JS popup dismisser (runs on every new page load
    # via CDP — no per-action overhead)
    inject_persistent_dismisser(driver)

    # Wrap the driver so lightweight popup check runs after get / click / submit
    driver = _install_popup_hooks(driver)

    return driver


def save_script(script_code: str) -> str:
    """Save the generated script to the outputs/generated_scripts directory.

    Returns the file path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_script_{ts}.py"
    filepath = os.path.join(SCRIPT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(script_code)
    logger.info("Generated script saved to %s", filepath)
    return filepath


def execute_script(
    script_code: str,
    driver: webdriver.Chrome,
    url: str,
) -> Dict[str, Any]:
    """Execute a generated Selenium test script.

    The script is expected to define a ``run_test(driver)`` function that
    returns a results dictionary.

    Parameters
    ----------
    script_code : str
        The full Python source of the generated test script.
    driver : webdriver.Chrome
        An initialised Chrome WebDriver.
    url : str
        The target URL (passed for context; the script should navigate itself).

    Returns
    -------
    dict
        Execution results with per-step status.
    """
    # Navigate to URL first and handle popups
    try:
        driver.get(url)
        handle_popups(driver)
    except Exception as exc:
        logger.error("Initial navigation/popup handling failed: %s", exc)

    # Save script to a temp file and import it
    script_path = save_script(script_code)

    results: Dict[str, Any] = {
        "steps": [],
        "overall_status": "ERROR",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "error": None,
        "page_html": None,
    }

    try:
        # Load script as module
        spec = importlib.util.spec_from_file_location("generated_test", script_path)
        module = importlib.util.module_from_spec(spec)

        # Inject paths so the script can find the screenshot dirs
        module.__dict__["SCREENSHOT_DIR"] = SCREENSHOT_DIR
        module.__dict__["SCREENSHOT_SUCCESS_DIR"] = SCREENSHOT_SUCCESS_DIR
        module.__dict__["SCREENSHOT_FAILURE_DIR"] = SCREENSHOT_FAILURE_DIR

        spec.loader.exec_module(module)

        if not hasattr(module, "run_test"):
            results["error"] = "Generated script missing run_test(driver) function"
            logger.error(results["error"])
            return results

        # Execute
        test_results = module.run_test(driver)

        if isinstance(test_results, dict):
            results.update(test_results)
        else:
            results["error"] = f"run_test returned {type(test_results).__name__} instead of dict"

    except Exception as exc:
        logger.error("Script execution error: %s", exc, exc_info=True)
        results["error"] = f"{type(exc).__name__}: {str(exc)}"
        results["overall_status"] = "ERROR"
        # Capture page HTML for self-healing
        try:
            results["page_html"] = driver.page_source
        except Exception:
            pass
        # Capture screenshot
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            driver.save_screenshot(os.path.join(SCREENSHOT_DIR, f"execution_error_{ts}.png"))
        except Exception:
            pass

    results["end_time"] = datetime.now().isoformat()

    # Collect page HTML for potential healing
    if not results.get("page_html"):
        try:
            results["page_html"] = driver.page_source
        except Exception:
            pass

    return results
