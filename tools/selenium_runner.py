"""Selenium runner utility.

Sets up a Chrome WebDriver with hardened options (download prefs, popup
hooks, isolated profile) and provides helpers for executing generated
test scripts.

Integrates the **Explorer agent** seamlessly: ``explore_current_page()``
runs after every navigation and after any click that changes the URL,
caching intelligence per URL so the same page is never analysed twice.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import logging
import os
import re as _re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from tools.popup_handler import (
    handle_popups,
    handle_popups_light,
    inject_persistent_dismisser,
)

logger = logging.getLogger(__name__)

# ── WebDriver Manager caching ───────────────────────────────────────
os.environ.setdefault("WDM_CHECK_DRIVER_VERSION", os.getenv("WDM_CHECK_DRIVER_VERSION", "false"))
os.environ.setdefault("WDM_CACHE_VALID_RANGE", os.getenv("WDM_CACHE_VALID_RANGE", "7"))

# ── Directories ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_DIR = os.path.join(BASE_DIR, "outputs", "screenshots")
SCREENSHOT_SUCCESS_DIR = os.path.join(SCREENSHOT_DIR, "success")
SCREENSHOT_FAILURE_DIR = os.path.join(SCREENSHOT_DIR, "failure")
SCRIPT_DIR = os.path.join(BASE_DIR, "outputs", "generated_scripts")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "outputs", "downloads")
PAGE_INTEL_DIR = os.path.join(BASE_DIR, "outputs", "page_intelligence")

for _d in (SCREENSHOT_SUCCESS_DIR, SCREENSHOT_FAILURE_DIR, SCRIPT_DIR, DOWNLOAD_DIR, PAGE_INTEL_DIR):
    os.makedirs(_d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Page Intelligence Cache — one entry per URL
# ═══════════════════════════════════════════════════════════════════
page_intel_cache: Dict[str, Dict[str, Any]] = {}


def _save_page_intel(url: str, intel: Dict[str, Any]) -> str:
    """Persist page intelligence to a JSON file and return the path."""
    from urllib.parse import urlparse

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = urlparse(url).netloc.replace(".", "_")[:40] or "page"
    path = os.path.join(PAGE_INTEL_DIR, f"intel_{ts}_{slug}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"url": url, "timestamp": ts, **intel}, f, indent=2, default=str)
    logger.info("Page intel saved → %s", path)
    return path


def _apply_intel_to_execution(driver, intel: Dict[str, Any]) -> None:
    """React to page intelligence findings inside the running driver."""
    # obstacles? dismiss popups
    obstacles = intel.get("obstacles") or []
    if obstacles:
        logger.info("Explorer detected obstacles — running popup handler")
        try:
            handle_popups(driver)
        except Exception:
            pass

    # loading_state == "loading" → explicit wait
    if intel.get("loading_state") == "loading":
        logger.info("Explorer: page still loading — waiting for readyState")
        try:
            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except Exception:
            pass

    # visible_errors → log as warnings
    for err in intel.get("visible_errors") or []:
        logger.warning("Explorer spotted visible error: %s", err)


# ═══════════════════════════════════════════════════════════════════
# explore_current_page — DOM + Vision intelligence from live driver
# ═══════════════════════════════════════════════════════════════════

def explore_current_page(driver, page_url: str) -> Dict[str, Any]:
    """Gather DOM + visual intelligence from the ALREADY OPEN driver.

    This never opens a new browser or tab.  It captures intel from the
    live page, optionally sends screenshots to the vision LLM, and
    caches the result keyed by URL.

    Returns a dict with full page intelligence.
    """
    # Skip if already cached for this exact URL
    if page_url in page_intel_cache:
        return page_intel_cache[page_url]

    intel: Dict[str, Any] = {
        "url": page_url,
        "capture_timestamp": datetime.now().isoformat(),
        "page_type": "unknown",
        "main_purpose": "",
        "navigation_type": "none",
        "content_density": "medium",
        "scroll_required": False,
        "visible_errors": [],
        "loading_state": "loaded",
        "notable_elements": [],
        "obstacles": [],
        "captcha_present": False,
        "captcha_type": "none",
        "two_factor_present": False,
        "file_upload_present": False,
        "interactive_elements_count": 0,
        "suspicious_elements": [],
        "color_scheme": "",
        "mobile_responsive": True,
        # DOM-only fields
        "page_title": "",
        "headings": [],
        "form_fields": [],
        "buttons": [],
        "links_count": 0,
        "iframes": [],
        "captcha_indicators": [],
        "tfa_indicators": [],
    }

    # ── Layer 1: DOM Intelligence (fast, no LLM) ───────────────────
    try:
        intel["page_title"] = driver.title or ""

        # Headings
        for tag in ("h1", "h2"):
            for el in driver.find_elements(By.TAG_NAME, tag)[:5]:
                try:
                    intel["headings"].append({"tag": tag, "text": el.text.strip()[:120]})
                except Exception:
                    pass

        # Form fields
        for inp in driver.find_elements(By.TAG_NAME, "input")[:30]:
            try:
                field = {
                    "type": inp.get_attribute("type") or "text",
                    "id": inp.get_attribute("id") or "",
                    "name": inp.get_attribute("name") or "",
                    "placeholder": inp.get_attribute("placeholder") or "",
                }
                fid = field["id"]
                if fid:
                    try:
                        lbl = driver.find_element(By.CSS_SELECTOR, f'label[for="{fid}"]')
                        field["label"] = lbl.text.strip()[:60]
                    except Exception:
                        field["label"] = ""
                intel["form_fields"].append(field)
            except Exception:
                pass

        # Buttons
        for btn in driver.find_elements(By.TAG_NAME, "button")[:15]:
            try:
                intel["buttons"].append({
                    "text": btn.text.strip()[:60],
                    "id": btn.get_attribute("id") or "",
                    "type": btn.get_attribute("type") or "",
                    "aria_label": btn.get_attribute("aria-label") or "",
                })
            except Exception:
                pass

        # Links count
        intel["links_count"] = len(driver.find_elements(By.TAG_NAME, "a"))

        # File inputs
        file_inputs = driver.find_elements(By.CSS_SELECTOR, 'input[type="file"]')
        intel["file_upload_present"] = len(file_inputs) > 0

        # Iframes
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for ifr in iframes[:5]:
            try:
                intel["iframes"].append({
                    "src": (ifr.get_attribute("src") or "")[:200],
                })
            except Exception:
                pass

        # Captcha indicators
        page_src = ""
        try:
            page_src = driver.page_source.lower()
        except Exception:
            pass
        captcha_keywords = ["recaptcha", "hcaptcha", "captcha", "g-recaptcha", "h-captcha"]
        for kw in captcha_keywords:
            if kw in page_src:
                intel["captcha_indicators"].append(kw)
                intel["captcha_present"] = True

        for ifr_info in intel["iframes"]:
            src = ifr_info.get("src", "").lower()
            if "recaptcha" in src or "hcaptcha" in src:
                intel["captcha_present"] = True
                intel["captcha_type"] = "checkbox"

        # 2FA indicators
        tfa_keywords = ["otp", "authenticator", "verification code", "two-factor",
                        "2fa", "totp", "one-time"]
        for kw in tfa_keywords:
            if kw in page_src:
                intel["tfa_indicators"].append(kw)
                intel["two_factor_present"] = True

        # Interactive elements count
        interactive = driver.find_elements(By.CSS_SELECTOR,
            "input, button, select, textarea, a[href], [onclick], [role='button']")
        intel["interactive_elements_count"] = len(interactive)

    except Exception as exc:
        logger.warning("DOM intel gathering error: %s", exc)

    # ── Layer 2: Visual Intelligence (screenshot + vision LLM) ─────
    screenshot_path = ""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ss_dir = os.path.join(SCREENSHOT_DIR, "explorer")
        os.makedirs(ss_dir, exist_ok=True)

        # Screenshot 1: top of page
        ss_path_1 = os.path.join(ss_dir, f"explore_top_{ts}.png")
        driver.save_screenshot(ss_path_1)
        with open(ss_path_1, "rb") as f:
            screenshot_b64_top = base64.b64encode(f.read()).decode("ascii")

        # Screenshot 2: scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.5)
        ss_path_2 = os.path.join(ss_dir, f"explore_bottom_{ts}.png")
        driver.save_screenshot(ss_path_2)
        with open(ss_path_2, "rb") as f:
            screenshot_b64_bottom = base64.b64encode(f.read()).decode("ascii")

        # Scroll back to top
        driver.execute_script("window.scrollTo(0, 0)")

        screenshot_path = ss_path_1

        # Send both screenshots to vision LLM
        from llm.factory import get_browser_use_llm
        from langchain_core.messages import HumanMessage

        vision_prompt = (
            "Analyze these webpage screenshots (top and bottom of page) and return ONLY valid JSON:\n"
            "{\n"
            '  "page_type": "login|form|listing|dashboard|checkout|upload|captcha|other",\n'
            '  "main_purpose": "one sentence",\n'
            '  "navigation_type": "sidebar|topbar|none",\n'
            '  "content_density": "sparse|medium|dense",\n'
            '  "scroll_required": true/false,\n'
            '  "visible_errors": ["list of error messages visible"],\n'
            '  "loading_state": "loaded|loading|error",\n'
            '  "notable_elements": ["list of important visible UI elements"],\n'
            '  "obstacles": ["any popups/overlays/banners blocking the page"],\n'
            '  "captcha_present": true/false,\n'
            '  "captcha_type": "none|checkbox|image_select|text_input",\n'
            '  "two_factor_present": true/false,\n'
            '  "file_upload_present": true/false,\n'
            '  "interactive_elements_count": number,\n'
            '  "suspicious_elements": ["any honeypots or anti-bot measures"],\n'
            '  "color_scheme": "dominant colors",\n'
            '  "mobile_responsive": true/false\n'
            "}"
        )

        content_parts: list = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64_top}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64_bottom}"}},
            {"type": "text", "text": vision_prompt},
        ]

        msg = HumanMessage(content=content_parts)
        vision_llm = get_browser_use_llm()
        response = vision_llm.invoke([msg])

        # Parse JSON from vision LLM
        raw = response.content.strip()
        vision_intel: Optional[Dict[str, Any]] = None
        if raw.startswith("{"):
            try:
                vision_intel = json.loads(raw)
            except json.JSONDecodeError:
                pass
        if vision_intel is None:
            for pat in [r"```json\s*\n?(.*?)```", r"```\s*\n?(.*?)```", r"\{[\s\S]*\}"]:
                m = _re.search(pat, raw, _re.DOTALL)
                if m:
                    try:
                        candidate = m.group(1) if m.lastindex else m.group(0)
                        vision_intel = json.loads(candidate.strip())
                        break
                    except (json.JSONDecodeError, IndexError):
                        continue

        if vision_intel and isinstance(vision_intel, dict):
            for k, v in vision_intel.items():
                if k in intel and v is not None:
                    intel[k] = v

    except Exception as exc:
        logger.info("Vision intel skipped: %s", exc)

    # Store screenshot path for report
    intel["screenshot_path"] = screenshot_path

    # ── Cache and persist ──────────────────────────────────────────
    page_intel_cache[page_url] = intel
    _save_page_intel(page_url, intel)

    logger.info(
        "PAGE_INTEL | url=%s | %s",
        page_url,
        json.dumps(intel, indent=2, default=str),
    )

    # ── Apply immediate actions ────────────────────────────────────
    _apply_intel_to_execution(driver, intel)

    return intel


def _get_intel_summary(url: str) -> str:
    """Return a short one-line intel summary for the given URL."""
    intel = page_intel_cache.get(url)
    if not intel:
        return ""
    parts: list = []
    parts.append(intel.get("page_type", "unknown"))
    if intel.get("obstacles"):
        parts.append(f"obstacles={len(intel['obstacles'])}")
    else:
        parts.append("No obstacles")
    if intel.get("captcha_present"):
        parts.append(f"captcha={intel.get('captcha_type', 'yes')}")
    else:
        parts.append("No captcha")
    ff = intel.get("form_fields", [])
    parts.append(f"{len(ff)} form fields")
    return "Page intel: " + " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Driver wrapper — popup + explorer hooks on navigate / click / submit
# ═══════════════════════════════════════════════════════════════════

def _install_popup_hooks(driver: webdriver.Chrome) -> webdriver.Chrome:
    """Monkey-patch *driver* so that a **lightweight** popup check runs
    after every action that may trigger a new page or overlay:

    * ``driver.get(url)`` — popup check + explorer auto-run
    * ``element.click()`` — popup + explorer if URL changed
    * ``element.submit()`` — popup check

    The heavy JS observer (installed via CDP) handles actual popup
    dismissal in the background.
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
        # Explorer: auto-explore after navigation
        try:
            explore_current_page(driver, driver.current_url)
        except Exception as exc:
            logger.debug("explore_current_page after get() failed: %s", exc)

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
            url_before = ""
            try:
                url_before = driver.current_url
            except Exception:
                pass

            _orig_click()

            try:
                handle_popups_light(driver)
            except Exception as exc:
                logger.debug("handle_popups_light after click() failed: %s", exc)

            # If URL changed, run explorer on new page
            try:
                url_after = driver.current_url
                if url_before and url_after and url_after != url_before:
                    explore_current_page(driver, url_after)
            except Exception as exc:
                logger.debug("explore_current_page after click() failed: %s", exc)

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

    logger.info("Popup hooks + explorer hooks installed on driver")
    return driver


# ═══════════════════════════════════════════════════════════════════
# Chrome driver creation
# ═══════════════════════════════════════════════════════════════════

def create_chrome_driver(headless: bool = True) -> webdriver.Chrome:
    """Create a hardened Chrome WebDriver for deterministic automation runs.

    Prevents password popups, data leak warnings, onboarding, and sync
    issues.  Includes download preferences for file-download tests.
    """
    options = Options()

    # --- Headless mode ---
    if headless:
        options.add_argument("--headless=new")

    # --- Isolated temporary profile ---
    user_data_dir = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={user_data_dir}")

    # --- Stable automation settings ---
    options.add_argument("--incognito")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-sync")
    options.add_argument("--disable-background-networking")
    options.add_argument("--metrics-recording-only")
    options.add_argument("--disable-component-update")
    options.add_argument("--disable-client-side-phishing-detection")

    # --- Disable password manager + leak detection ---
    options.add_argument(
        "--disable-features="
        "PasswordLeakDetection,"
        "PasswordManagerOnboarding,"
        "AutofillServerCommunication,"
        "SafeBrowsingEnhancedProtection"
    )

    # --- Core stability flags ---
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--window-size=1920,1080")

    # --- Reduce automation fingerprints ---
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-blink-features=AutomationControlled")

    # --- Clean preferences + download/upload directory ---
    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "profile.password_manager_leak_detection": False,
        "profile.default_content_setting_values.notifications": 2,
        "safebrowsing.enabled": False,
        "safebrowsing.disable_download_protection": True,
        "download.default_directory": str(DOWNLOAD_DIR),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
    }
    options.add_experimental_option("prefs", prefs)

    # --- Create driver ---
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception:
        driver = webdriver.Chrome(options=options)

    # --- Hide navigator.webdriver ---
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """
        },
    )

    driver.set_page_load_timeout(60)
    driver.implicitly_wait(5)

    # Clean up temp profile when quitting
    original_quit = driver.quit

    def custom_quit():
        original_quit()
        shutil.rmtree(user_data_dir, ignore_errors=True)

    driver.quit = custom_quit

    return driver


# ═══════════════════════════════════════════════════════════════════
# File upload handler (Selenium only — no Browser Use needed)
# ═══════════════════════════════════════════════════════════════════

def handle_file_upload(driver: webdriver.Chrome, step: Dict[str, Any]) -> Dict[str, Any]:
    """Upload a file using Selenium ``send_keys()`` on an ``<input type='file'>``.

    Tries multiple locator strategies.  Returns a step-result dict.
    """
    file_path = step.get("value", "")
    target = step.get("target", "file upload element")

    if not file_path or not os.path.isfile(file_path):
        return {
            "step": step.get("step", 0),
            "action": "upload_file",
            "target": target,
            "status": "FAILED",
            "error": f"File not found: {file_path}",
            "screenshot": "",
            "locator_used": "",
            "timestamp": datetime.now().isoformat(),
            "engine": "selenium",
        }

    strategies = [
        (By.CSS_SELECTOR, "input[type='file']"),
        (By.XPATH, "//input[@type='file']"),
        (By.CSS_SELECTOR, "[class*='upload'] input"),
        (By.CSS_SELECTOR, "[id*='upload'] input"),
    ]

    uploaded = False
    locator_used = ""
    for by, selector in strategies:
        try:
            file_input = driver.find_element(by, selector)
            file_input.send_keys(os.path.abspath(file_path))
            locator_used = f"{by}: {selector}"
            uploaded = True
            logger.info("File uploaded via %s — %s", locator_used, os.path.basename(file_path))
            break
        except Exception:
            continue

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if uploaded:
        ss = os.path.join(SCREENSHOT_SUCCESS_DIR, f"upload_step{step.get('step', 0)}_{ts}.png")
    else:
        ss = os.path.join(SCREENSHOT_FAILURE_DIR, f"upload_step{step.get('step', 0)}_{ts}.png")
    try:
        driver.save_screenshot(ss)
    except Exception:
        ss = ""

    if uploaded:
        return {
            "step": step.get("step", 0),
            "action": "upload_file",
            "target": target,
            "status": "PASSED",
            "error": "",
            "screenshot": ss,
            "locator_used": locator_used,
            "timestamp": datetime.now().isoformat(),
            "engine": "selenium",
        }
    else:
        return {
            "step": step.get("step", 0),
            "action": "upload_file",
            "target": target,
            "status": "FAILED",
            "error": "Could not find a file input element with any strategy",
            "screenshot": ss,
            "locator_used": "",
            "timestamp": datetime.now().isoformat(),
            "engine": "selenium",
        }


# ═══════════════════════════════════════════════════════════════════
# Script save helper
# ═══════════════════════════════════════════════════════════════════

def save_script(script_code: str) -> str:
    """Save the generated script to outputs/generated_scripts/.  Returns path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_script_{ts}.py"
    filepath = os.path.join(SCRIPT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(script_code)
    logger.info("Generated script saved to %s", filepath)
    return filepath


# ═══════════════════════════════════════════════════════════════════
# Script execution
# ═══════════════════════════════════════════════════════════════════

def execute_script(
    script_code: str,
    driver: webdriver.Chrome,
    url: str,
) -> Dict[str, Any]:
    """Execute a generated Selenium test script.

    The script is expected to define a ``run_test(driver)`` function that
    returns a results dictionary.

    **Fix 6 guarantee**: if ``run_test()`` raises an unhandled exception
    the overall status is always ``ERROR`` — never ``PASSED``.
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
            if not results.get("steps") and not results.get("error"):
                results["overall_status"] = "ERROR"
                results["error"] = "run_test returned no steps"
        else:
            results["error"] = f"run_test returned {type(test_results).__name__} instead of dict"
            results["overall_status"] = "ERROR"

    except Exception as exc:
        # ── Fix 6: NEVER report PASSED on script crash ─────────────
        logger.error("Script execution error: %s", exc, exc_info=True)
        results["error"] = f"{type(exc).__name__}: {str(exc)}"
        results["overall_status"] = "ERROR"
        results["steps"] = [{
            "step": 0,
            "action": "script_execution",
            "target": "",
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {str(exc)}",
            "error_type": type(exc).__name__,
            "screenshot": "",
            "locator_used": "",
            "timestamp": datetime.now().isoformat(),
        }]
        try:
            results["page_html"] = driver.page_source
        except Exception:
            pass
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            driver.save_screenshot(os.path.join(SCREENSHOT_DIR, f"execution_error_{ts}.png"))
        except Exception:
            pass

    results["end_time"] = datetime.now().isoformat()

    # ── Fix 6: Final guard — override PASSED when an error occurred ──
    if results.get("error") and results.get("overall_status") == "PASSED":
        results["overall_status"] = "ERROR"

    # Collect page HTML for potential healing
    if not results.get("page_html"):
        try:
            results["page_html"] = driver.page_source
        except Exception:
            pass

    # ── Attach per-step intel summary ──────────────────────────────
    for step in results.get("steps", []):
        # ── Bug 2 fix: ensure every failed step has failed_url ─────
        if step.get("status") == "FAILED" and not step.get("failed_url"):
            try:
                step["failed_url"] = driver.current_url
            except Exception:
                step["failed_url"] = url
        try:
            step_url = step.get("failed_url") or step.get("url_on_failure") or url
            step["intel_summary"] = _get_intel_summary(step_url) or _get_intel_summary(url)
        except Exception:
            step["intel_summary"] = ""

    # ── Attach all page intel to results ───────────────────────────
    results["page_intelligence_cache"] = dict(page_intel_cache)

    return results
