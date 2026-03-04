"""Browser Use Runner — handles complex steps via Browser Use + Playwright.

Supports two complexity types routed here:
* **captcha** — attempts to solve a captcha with a vision-capable LLM
* **two_factor** — enters a TOTP code (auto-generated or manual entry)

File uploads are handled by Selenium directly (see selenium_runner.py).

Every public function returns a *StepResult* dict with the exact same keys
that ``selenium_runner`` produces, so the rest of the pipeline can consume
it transparently.
"""

from __future__ import annotations

# ── Fix 4: Windows asyncio — ProactorEventLoop for Playwright ──────
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_DIR = os.path.join(BASE_DIR, "outputs", "screenshots", "browser_use")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ── Max retry attempts for Browser Use agent ───────────────────────
_MAX_RETRIES = 3


def _get_browser_use_native_llm():
    """Return a Browser Use *native* LLM wrapper.

    Uses the classes defined in ``browser_use.llm`` — NOT LangChain's
    ``ChatOpenAI``.  Browser Use's ``Agent`` expects objects that satisfy
    ``browser_use.llm.BaseChatModel`` (provider property, async ainvoke,
    etc.) and these native wrappers are the only ones guaranteed to work.

    Supported providers (via BROWSER_USE_PROVIDER env var):
    * ``groq``     → ``browser_use.llm.ChatGroq``
    * ``cerebras`` → ``browser_use.llm.ChatCerebras``
    * ``openai``   → ``browser_use.llm.ChatOpenAI``  (NOT langchain's!)
    """
    provider = os.getenv("BROWSER_USE_PROVIDER", "groq").lower().strip()

    if provider == "groq":
        from browser_use.llm import ChatGroq

        llm = ChatGroq(
            model=os.getenv(
                "BROWSER_USE_MODEL",
                "meta-llama/llama-4-scout-17b-16e-instruct",
            ),
            api_key=os.getenv(
                "BROWSER_USE_API_KEY", os.getenv("GROQ_API_KEY", "")
            ),
            temperature=0.1,
        )
        logger.info("Browser Use LLM: ChatGroq model=%s", llm.model)
        return llm

    if provider == "cerebras":
        from browser_use.llm import ChatCerebras

        llm = ChatCerebras(
            model=os.getenv("BROWSER_USE_MODEL", "llama-3.3-70b"),
            api_key=os.getenv(
                "BROWSER_USE_API_KEY", os.getenv("CEREBRAS_API_KEY", "")
            ),
            temperature=0.1,
        )
        logger.info("Browser Use LLM: ChatCerebras model=%s", llm.model)
        return llm

    if provider == "openai":
        from browser_use.llm import ChatOpenAI as BUChatOpenAI

        llm = BUChatOpenAI(
            model=os.getenv("BROWSER_USE_MODEL", "gpt-4o"),
            api_key=os.getenv(
                "BROWSER_USE_API_KEY", os.getenv("OPENAI_API_KEY", "")
            ),
            base_url=os.getenv("BROWSER_USE_BASE_URL") or None,
            temperature=0.1,
        )
        logger.info("Browser Use LLM: ChatOpenAI model=%s", llm.model)
        return llm

    # Fallback — treat as OpenAI-compatible with custom base_url
    from browser_use.llm import ChatOpenAI as BUChatOpenAI

    llm = BUChatOpenAI(
        model=os.getenv(
            "BROWSER_USE_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
        api_key=os.getenv(
            "BROWSER_USE_API_KEY", os.getenv("GROQ_API_KEY", "")
        ),
        base_url=os.getenv(
            "BROWSER_USE_BASE_URL", "https://api.groq.com/openai/v1"
        ),
        temperature=0.1,
    )
    logger.info(
        "Browser Use LLM: ChatOpenAI (fallback) model=%s base_url=%s",
        llm.model, llm.base_url,
    )
    return llm


# ── StepResult helper ──────────────────────────────────────────────
def _make_result(
    step: Dict[str, Any],
    status: str = "PASSED",
    error: str = "",
    screenshot: str = "",
    locator_used: str = "",
    current_url: str = "",
    cookies: list | None = None,
    browser_use_handled: bool = False,
) -> Dict[str, Any]:
    """Build a StepResult dict matching selenium_runner's format."""
    return {
        "step": step.get("step", 0),
        "action": step.get("action", ""),
        "target": step.get("target", ""),
        "status": status,
        "error": error,
        "screenshot": screenshot,
        "locator_used": locator_used,
        "timestamp": datetime.now().isoformat(),
        "engine": "browser_use",
        "current_url": current_url,
        "cookies": cookies or [],
        "browser_use_handled": browser_use_handled,
    }


def _screenshot_path(step: Dict[str, Any], tag: str = "") -> str:
    """Generate a timestamped screenshot file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_num = step.get("step", 0)
    return os.path.join(SCREENSHOT_DIR, f"step{step_num}_{tag}_{ts}.png")


# ── Async core running helper (Fix 4 — ProactorEventLoop) ─────────
def _run_async(coro):
    """Run an async coroutine from synchronous code safely.

    On Windows uses ``ProactorEventLoop`` explicitly to avoid
    ``NotImplementedError`` from Playwright.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(_run_in_proactor, coro).result()
    else:
        return _run_in_proactor(coro)


def _run_in_proactor(coro):
    """Create a ProactorEventLoop (Windows) or default loop and run."""
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    else:
        return asyncio.run(coro)


# ── Screenshot extraction helper ──────────────────────────────────
def _extract_screenshot(history) -> str:
    """Try to extract a screenshot from the last Browser Use history item."""
    if not history or not history.history:
        return ""
    try:
        last = history.history[-1]
        if hasattr(last, "state") and last.state:
            screenshot_b64 = getattr(last.state, "screenshot", None)
            if screenshot_b64:
                import base64
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SCREENSHOT_DIR, f"bu_agent_{ts}.png")
                with open(path, "wb") as f:
                    f.write(base64.b64decode(screenshot_b64))
                return path
    except Exception as exc:
        logger.debug("Screenshot extraction failed: %s", exc)
    return ""


# ── Browser state extraction helper ────────────────────────────────
async def _extract_browser_state(agent, browser, history, fallback_url: str):
    """Extract current URL and cookies from the live Browser Use session.

    Tries multiple access patterns and falls back gracefully.

    Returns ``(current_url, cookies)``.
    """
    current_url = fallback_url
    cookies: list = []

    # URL from last history state
    try:
        if history and history.history:
            for item in reversed(history.history):
                state = getattr(item, "state", None)
                if state:
                    u = getattr(state, "url", "")
                    if u:
                        current_url = u
                        break
    except Exception:
        pass

    # Playwright BrowserContext (for cookies)
    pw_context = None

    try:
        bc = getattr(agent, "browser_context", None)
        if bc:
            for attr in ("context", "_context"):
                ctx = getattr(bc, attr, None)
                if ctx and hasattr(ctx, "cookies"):
                    pw_context = ctx
                    break
            if pw_context is None:
                session = getattr(bc, "session", None)
                if session:
                    ctx = getattr(session, "context", None)
                    if ctx and hasattr(ctx, "cookies"):
                        pw_context = ctx
    except Exception:
        pass

    if pw_context is None:
        try:
            for attr in ("playwright_browser", "_playwright_browser", "_browser"):
                pw_browser = getattr(browser, attr, None)
                if pw_browser and hasattr(pw_browser, "contexts"):
                    ctxs = pw_browser.contexts
                    if ctxs:
                        pw_context = ctxs[0]
                        break
        except Exception:
            pass

    if pw_context:
        try:
            cookies = await pw_context.cookies()
        except Exception as exc:
            logger.debug("Cookie extraction failed: %s", exc)
        try:
            if pw_context.pages and current_url == fallback_url:
                current_url = pw_context.pages[-1].url
        except Exception:
            pass

    logger.info(
        "Extracted Browser Use state: url=%s  cookies=%d",
        current_url, len(cookies),
    )
    return current_url, cookies


# ═══════════════════════════════════════════════════════════════════
# Browser Use agent helper (with retry — Fix 5)
# ═══════════════════════════════════════════════════════════════════
async def _run_browser_use_agent(
    task: str,
    url: str,
    use_vision: bool = True,
    max_steps: int = 10,
    available_file_paths: Optional[list] = None,
    is_captcha: bool = False,
) -> dict:
    """Launch a Browser Use ``Agent``, run the task with up to 3 retries.

    Returns ``{"success": bool, "result": str, "screenshot": str, ...}``.
    """
    from browser_use import Agent, Browser

    llm = _get_browser_use_native_llm()
    browser = Browser(headless=False)

    last_error = ""
    screenshot_path = ""

    try:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                agent_kwargs: dict = dict(
                    task=task,
                    llm=llm,
                    browser=browser,
                    use_vision=use_vision,
                    max_actions_per_step=3,
                )
                if available_file_paths:
                    agent_kwargs["available_file_paths"] = available_file_paths

                agent = Agent(**agent_kwargs)
                history = await agent.run(max_steps=max_steps)

                screenshot_path = _extract_screenshot(history)

                final_result = history.final_result() if history else ""
                is_done = history.is_done() if history else False

                if is_done:
                    cur_url, cur_cookies = await _extract_browser_state(
                        agent, browser, history, "",
                    )
                    return {
                        "success": True,
                        "result": final_result or "",
                        "screenshot": screenshot_path,
                        "current_url": cur_url,
                        "cookies": cur_cookies,
                    }

                logger.warning(
                    "Browser Use attempt %d/%d did not complete (is_done=%s)",
                    attempt, _MAX_RETRIES, is_done,
                )
                last_error = final_result or "Agent did not complete the task"

            except Exception as exc:
                logger.warning(
                    "Browser Use attempt %d/%d error: %s",
                    attempt, _MAX_RETRIES, exc,
                )
                last_error = str(exc)

            # Between retries for captcha — try to refresh the captcha
            if is_captcha and attempt < _MAX_RETRIES:
                try:
                    refresh_agent = Agent(
                        task=(
                            "Look for a refresh/reload captcha button on this page "
                            "and click it to get a new captcha image. "
                            "If there is no such button, just report done."
                        ),
                        llm=llm,
                        browser=browser,
                        use_vision=True,
                        max_actions_per_step=2,
                    )
                    await refresh_agent.run(max_steps=3)
                except Exception:
                    pass

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(1)

        # All retries exhausted
        return {
            "success": False,
            "result": last_error,
            "screenshot": screenshot_path,
            "current_url": "",
            "cookies": [],
        }

    finally:
        try:
            await browser.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Captcha-type-aware prompt builder
# ═══════════════════════════════════════════════════════════════════

def build_captcha_task(url: str, captcha_type: str) -> str:
    """Return a Browser Use task prompt tailored to the captcha type.

    Parameters
    ----------
    url : str
        The page URL where the captcha is displayed.
    captcha_type : str
        One of: ``image_text``, ``checkbox``, ``image_select``,
        ``slider``, ``text_input``, ``none``, or any unknown value.
    """
    captcha_type = (captcha_type or "none").lower().strip()

    common_footer = (
        "\n\nIMPORTANT — avoid infinite loops:\n"
        "- If you have already tried an answer and it was wrong, do NOT "
        "enter the same answer again.\n"
        "- After 3 failed attempts, stop and report done with success=False.\n"
        "  Do NOT keep clicking Reset indefinitely."
    )

    if captcha_type in ("image_text", "text_input"):
        return (
            f"Go to {url}.\n"
            "A text-based captcha image is displayed. Follow these steps:\n"
            "1. Look at the captcha image character by character, left to right.\n"
            "2. Watch for common character confusions:\n"
            "   - 0 (zero) vs O (letter O)\n"
            "   - 1 (one) vs l (lowercase L) vs I (uppercase i)\n"
            "   - 5 vs S, 8 vs B, 6 vs G\n"
            "   - rn vs m, cl vs d, vv vs w\n"
            "3. Type the EXACT captcha text, preserving uppercase/lowercase.\n"
            "4. Click the submit/verify button.\n"
            "5. If the answer is wrong, click the refresh/reload captcha button "
            "to get a new image, read it again carefully, and submit a "
            "COMPLETELY DIFFERENT answer.\n"
            "6. Report success ONLY when the page confirms the captcha is correct."
            f"{common_footer}"
        )

    if captcha_type == "checkbox":
        return (
            f"Go to {url}.\n"
            "A checkbox captcha (e.g. reCAPTCHA 'I am not a robot') is present.\n"
            "1. Locate the checkbox inside the captcha iframe or widget.\n"
            "2. Click the checkbox ONCE.\n"
            "3. Wait 2-3 seconds for verification.\n"
            "4. If an image challenge grid appears after clicking:\n"
            "   a. Read the instruction carefully (e.g. 'Select all images "
            "with traffic lights').\n"
            "   b. Click every matching tile.\n"
            "   c. Click the Verify / Next button.\n"
            "   d. Repeat if a new round appears.\n"
            "5. Confirm the green checkmark appears, then report success."
            f"{common_footer}"
        )

    if captcha_type == "image_select":
        return (
            f"Go to {url}.\n"
            "An image-selection captcha is displayed (grid of image tiles).\n"
            "1. Read the instruction text VERY carefully (e.g. 'Select all "
            "images with bicycles').\n"
            "2. Examine EACH tile individually.\n"
            "3. Click every tile that matches the instruction.\n"
            "4. Click the Verify / Submit button.\n"
            "5. If a new round of images appears, repeat the process.\n"
            "6. Report success ONLY when the page confirms the challenge "
            "is solved."
            f"{common_footer}"
        )

    if captcha_type == "slider":
        return (
            f"Go to {url}.\n"
            "A slider captcha is present (drag a handle to align a puzzle "
            "piece).\n"
            "1. Locate the slider handle element.\n"
            "2. Click and HOLD the handle.\n"
            "3. Drag it slowly to the right until the puzzle piece aligns "
            "with the gap.\n"
            "4. Release the mouse button.\n"
            "5. If verification fails, try again with a slightly different "
            "drag speed or distance.\n"
            "6. Report success when the page indicates the slider was solved "
            "correctly."
            f"{common_footer}"
        )

    # "none" or unknown — generic fallback
    return (
        f"Go to {url}.\n"
        "There may be a verification challenge or captcha on this page.\n"
        "1. Inspect the page for any captcha widget, verification checkbox, "
        "image challenge, slider, or distorted text input.\n"
        "2. Solve the challenge using the appropriate interaction method.\n"
        "3. If it is a text captcha, read character by character and watch "
        "for confusions (0/O, 1/l/I, 5/S, 8/B).\n"
        "4. If it is a checkbox, click it and handle any follow-up image "
        "challenge.\n"
        "5. Report success when the page confirms the challenge is solved."
        f"{common_footer}"
    )


# ═══════════════════════════════════════════════════════════════════
# CAPTCHA handler (captcha-type-aware)
# ═══════════════════════════════════════════════════════════════════
async def _handle_captcha(step: Dict[str, Any], url: str) -> Dict[str, Any]:
    """Attempt to solve a captcha using Browser Use with vision + retries."""
    captcha_type = step.get("captcha_type", "none")
    task = build_captcha_task(url, captcha_type)

    logger.info("Captcha handler: type=%s  url=%s", captcha_type, url)

    try:
        result = await _run_browser_use_agent(
            task=task,
            url=url,
            use_vision=True,
            max_steps=8,
            is_captcha=True,
        )

        return _make_result(
            step,
            status="PASSED" if result["success"] else "FAILED",
            error="" if result["success"] else f"Captcha solving unsuccessful: {result.get('result', '')}",
            screenshot=result.get("screenshot", ""),
            current_url=result.get("current_url", ""),
            cookies=result.get("cookies", []),
            browser_use_handled=result["success"],
        )

    except Exception as exc:
        logger.error("Captcha handling failed: %s", exc)
        return _make_result(
            step,
            status="FAILED",
            error=f"Captcha handling error: {exc}",
        )


# ═══════════════════════════════════════════════════════════════════
# TWO-FACTOR (TOTP) handler
# ═══════════════════════════════════════════════════════════════════
async def _handle_two_factor(step: Dict[str, Any], url: str) -> Dict[str, Any]:
    """Enter a TOTP code — auto-generated from TOTP_SECRET or manual input."""
    totp_secret = os.getenv("TOTP_SECRET", "").strip()
    target = step.get("target", "OTP input field")

    if totp_secret:
        import pyotp
        code = pyotp.TOTP(totp_secret).now()
        logger.info("Generated TOTP code: %s", code)
    else:
        logger.info("No TOTP_SECRET set — prompting user for code")
        print("\n" + "=" * 50)
        print("  2FA VERIFICATION REQUIRED")
        print("  No TOTP_SECRET found in .env")
        print("=" * 50)
        code = input("  Enter the 2FA code: ").strip()
        print("=" * 50 + "\n")

    if not code:
        return _make_result(
            step,
            status="FAILED",
            error="No TOTP code available (no secret and no manual input)",
        )

    # Use Playwright to type the code
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)

            otp_selectors = [
                'input[name*="otp"]',
                'input[name*="code"]',
                'input[name*="token"]',
                'input[name*="totp"]',
                'input[name*="2fa"]',
                'input[name*="verification"]',
                'input[autocomplete="one-time-code"]',
                'input[type="tel"]',
                'input[type="number"][maxlength="6"]',
                'input[inputmode="numeric"]',
            ]

            filled = False
            for selector in otp_selectors:
                try:
                    el = page.locator(selector).first
                    if await el.is_visible(timeout=1000):
                        await el.fill(code)
                        filled = True
                        logger.info("Filled OTP code using selector: %s", selector)
                        break
                except Exception:
                    continue

            if not filled:
                logger.warning("Could not find OTP input via selectors, trying Browser Use agent")
                await browser.close()

                bu_task = (
                    f"Go to {url}. "
                    f"Find the input field for 2FA/OTP/verification code "
                    f"(described as '{target}') and type the code: {code}"
                )

                bu_result = await _run_browser_use_agent(
                    task=bu_task,
                    url=url,
                    use_vision=True,
                    max_steps=10,
                )

                return _make_result(
                    step,
                    status="PASSED" if bu_result["success"] else "FAILED",
                    error="" if bu_result["success"] else f"2FA code entry failed: {bu_result.get('result', '')}",
                    screenshot=bu_result.get("screenshot", ""),
                    current_url=bu_result.get("current_url", ""),
                    cookies=bu_result.get("cookies", []),
                    browser_use_handled=bu_result["success"],
                )

            await page.wait_for_timeout(2000)
            ss_path = _screenshot_path(step, "2fa")
            await page.screenshot(path=ss_path)
            await browser.close()

        return _make_result(step, status="PASSED", screenshot=ss_path)

    except Exception as exc:
        logger.error("2FA handling failed: %s", exc)
        return _make_result(
            step,
            status="FAILED",
            error=f"2FA handling error: {exc}",
        )


# ═══════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════
def execute_step(step: Dict[str, Any], url: str = "") -> Dict[str, Any]:
    """Execute a single complex step via Browser Use / Playwright.

    Dispatches to the appropriate async handler based on ``step["complexity"]``.
    Blocks until complete.
    """
    complexity = step.get("complexity", "standard")

    handler_map = {
        "captcha": _handle_captcha,
        "two_factor": _handle_two_factor,
    }

    handler = handler_map.get(complexity)
    if not handler:
        return _make_result(
            step,
            status="FAILED",
            error=f"Unknown complexity type for Browser Use: {complexity}",
        )

    try:
        return _run_async(handler(step, url))
    except Exception as exc:
        logger.error("browser_use_runner.execute_step crashed: %s", exc, exc_info=True)
        return _make_result(
            step,
            status="FAILED",
            error=f"Browser Use runner crash: {exc}",
        )
