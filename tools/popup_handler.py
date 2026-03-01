"""Popup / overlay handler utility — performance-optimised.

Uses a **two-tier** strategy so the handler stays dormant when no popup
is present and avoids unnecessary DOM scanning on every action.

Tier 1 — Persistent JS observer (browser-side, zero Python round-trips):
    Injected **once** at driver creation via Chrome DevTools Protocol
    (``Page.addScriptToEvaluateOnNewDocument``).  Runs automatically on
    every new document.  A debounced ``MutationObserver`` with
    ``requestIdleCallback`` sweeps dismiss-buttons only when new DOM
    nodes appear — not on every attribute change.

Tier 2 — Gated Python fallback (for stubborn popups):
    ``handle_popups()`` first runs a **fast JS overlay-detection gate**
    (~2–5 ms).  Only when an overlay is actually visible does it execute
    the more expensive Python-level selector iteration.  Time-debounced
    at 1 second so rapid consecutive calls are skipped.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List

from selenium.common.exceptions import (
    NoAlertPresentException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

logger = logging.getLogger(__name__)

# ── Debounce state ──────────────────────────────────────────────────
_last_full_sweep: float = 0.0
_DEBOUNCE_SEC: float = 1.0   # skip repeated full sweeps within 1 s

# ═══════════════════════════════════════════════════════════════════
# Tier 1 — Persistent JavaScript observer
# ═══════════════════════════════════════════════════════════════════
# Kept as POPUP_DISMISSER_JS for backward compatibility (tests import it).

POPUP_DISMISSER_JS = r"""
(function() {
    'use strict';
    if (window.__popupDismisserActive) return;
    window.__popupDismisserActive = true;

    var SELS = [
        '[id*="cookie"] button','[class*="cookie"] button',
        '[id*="consent"] button','[class*="consent"] button',
        '[id*="gdpr"] button','[class*="gdpr"] button',
        '[aria-label*="cookie" i]','[aria-label*="consent" i]',
        '[aria-label*="close" i]','[aria-label*="dismiss" i]',
        '[aria-label*="accept" i]',
        'button[class*="close"]','button[class*="dismiss"]',
        'button[class*="accept"]','a[class*="close"]',
        '.modal .close','.overlay .close',
        '[data-dismiss="modal"]','[data-action="close"]',
        '#onetrust-accept-btn-handler',
        '.cc-btn.cc-dismiss','.cc-btn.cc-allow',
        '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
        '.cky-btn-accept','#accept-cookies',
        '#cookie-accept','.cookie-accept',
        'button[data-cookiefirst-action="accept"]',
        '.js-cookie-consent-agree',
        '.popup-close','.newsletter-close',
        '[class*="popup"] [class*="close"]',
        '[class*="newsletter"] button[class*="close"]',
        '.modal-close','.modal__close'
    ];

    var DW = /^(ok|close|accept|agree|dismiss|allow|got it|continue|skip|no thanks|not now|i agree|i accept|allow all|accept all|reject all|deny|later|maybe later|remind me later)$/i;

    function isOvl(el) {
        var n = el;
        while (n && n !== document.body) {
            try {
                var c = getComputedStyle(n);
                if (c.position === 'fixed' || c.position === 'sticky') return true;
                if (parseInt(c.zIndex, 10) > 999) return true;
                if (n.getAttribute('role') === 'dialog' ||
                    n.getAttribute('aria-modal') === 'true') return true;
            } catch(e) {}
            n = n.parentElement;
        }
        return false;
    }

    function sweep() {
        for (var i = 0; i < SELS.length; i++) {
            try {
                var els = document.querySelectorAll(SELS[i]);
                for (var j = 0; j < els.length; j++) {
                    if (els[j].offsetParent !== null) els[j].click();
                }
            } catch(e) {}
        }
        var bs = document.querySelectorAll('button, a, [role="button"]');
        for (var k = 0; k < bs.length; k++) {
            try {
                var b = bs[k];
                if (b.offsetParent === null) continue;
                var t = (b.innerText || b.textContent || '').trim();
                if (t.length > 40) continue;
                if (DW.test(t) && isOvl(b)) b.click();
            } catch(e) {}
        }
    }

    // Initial sweep + single delayed retry for late-rendering popups
    sweep();
    setTimeout(sweep, 2000);

    // Debounced MutationObserver — fires ONLY when new nodes are added
    var tid = null;
    var sched = window.requestIdleCallback || function(f) { return setTimeout(f, 150); };
    var obs = new MutationObserver(function(muts) {
        for (var m = 0; m < muts.length; m++) {
            if (muts[m].addedNodes.length > 0) {
                if (!tid) { tid = sched(function() { tid = null; sweep(); }); }
                return;
            }
        }
    });
    if (document.body) {
        obs.observe(document.body, { childList: true, subtree: true });
    }
    // Auto-disconnect after 30 s to free resources
    setTimeout(function() { obs.disconnect(); window.__popupDismisserActive = false; }, 30000);
})();
"""

# Convenience alias
PERSISTENT_DISMISSER_JS = POPUP_DISMISSER_JS


# ── Fast overlay-detection gate (~2–5 ms) ───────────────────────────
# Returns true ONLY if a popup-like overlay is currently visible.

_OVERLAY_DETECT_JS = """
return (function() {
    var q = '[role="dialog"]:not([hidden]),[aria-modal="true"]:not([hidden]),' +
            '#onetrust-banner-sdk,#CybotCookiebotDialog,.cky-consent-container';
    var hit = document.querySelector(q);
    if (hit) {
        try {
            var s = getComputedStyle(hit);
            if (s.display !== 'none' && s.visibility !== 'hidden') return true;
        } catch(e) {}
    }
    var cands = document.querySelectorAll(
        '[class*="modal"],[class*="popup"],[class*="overlay"],' +
        '[class*="cookie"],[class*="consent"],[class*="gdpr"]');
    for (var i = 0; i < cands.length; i++) {
        try {
            var el = cands[i], cs = getComputedStyle(el);
            if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
            if ((cs.position === 'fixed' || cs.position === 'sticky' ||
                 parseInt(cs.zIndex, 10) > 999) && el.offsetWidth > 0 && el.offsetHeight > 0)
                return true;
        } catch(e) {}
    }
    return false;
})();
"""


# ═══════════════════════════════════════════════════════════════════
# Tier 1 helpers
# ═══════════════════════════════════════════════════════════════════

def inject_persistent_dismisser(driver: WebDriver) -> None:
    """Install the JS popup dismisser to auto-run on every new page load.

    Call this **once** during driver creation.  Uses Chrome DevTools
    Protocol so the script persists across navigations without
    re-injection.
    """
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": PERSISTENT_DISMISSER_JS},
        )
        logger.info("Persistent popup dismisser installed via CDP")
    except Exception as exc:
        logger.warning("CDP popup-dismisser injection failed: %s", exc)


def _ensure_js_dismisser(driver: WebDriver) -> None:
    """Inject the JS dismisser if not already active on this page.

    Fallback for environments where CDP injection is unavailable.
    Costs one tiny ``execute_script`` round-trip (~1 ms).
    """
    try:
        if not driver.execute_script("return !!window.__popupDismisserActive;"):
            driver.execute_script(PERSISTENT_DISMISSER_JS)
    except Exception:
        pass


def _has_visible_overlay(driver: WebDriver) -> bool:
    """Fast JS check — returns True only if a popup-like overlay is
    currently visible.  Typically completes in 2–5 ms."""
    try:
        return bool(driver.execute_script(_OVERLAY_DETECT_JS))
    except Exception:
        return False


# Legacy helper — kept for backward compat but now just delegates
def inject_popup_dismisser(driver: WebDriver) -> None:
    """Inject the JavaScript popup dismisser into the current page.

    .. deprecated:: Use :func:`inject_persistent_dismisser` (once at
       driver creation) instead.
    """
    _ensure_js_dismisser(driver)


# ═══════════════════════════════════════════════════════════════════
# Tier 2 — Python-level fallback (only runs when overlay detected)
# ═══════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Native browser dialog handler
# ---------------------------------------------------------------------------

def dismiss_browser_dialogs(driver: WebDriver) -> int:
    """Accept any pending native Chrome alert / confirm / prompt.

    Single attempt — returns 0 or 1.
    """
    try:
        alert = driver.switch_to.alert
        alert_text = alert.text
        alert.accept()
        logger.info("Dismissed native alert: %s", alert_text[:120])
        return 1
    except (NoAlertPresentException, WebDriverException):
        return 0


# ---------------------------------------------------------------------------
# Hardcoded-selector Python dismisser
# ---------------------------------------------------------------------------

_CLOSE_BUTTON_SELECTORS = [
    (By.CSS_SELECTOR, "#onetrust-accept-btn-handler"),
    (By.CSS_SELECTOR, ".cc-btn.cc-dismiss"),
    (By.CSS_SELECTOR, ".cc-btn.cc-allow"),
    (By.CSS_SELECTOR, "[data-dismiss='modal']"),
    (By.CSS_SELECTOR, "button[class*='close']"),
    (By.CSS_SELECTOR, "[aria-label='Close']"),
    (By.CSS_SELECTOR, "[aria-label='close']"),
    (By.CSS_SELECTOR, ".cookie-accept"),
    (By.CSS_SELECTOR, "#accept-cookies"),
    (By.CSS_SELECTOR, ".cky-btn-accept"),
    (By.XPATH, "//button[contains(translate(text(),'ACCEPT','accept'),'accept')]"),
    (By.XPATH, "//button[contains(translate(text(),'CLOSE','close'),'close')]"),
]


def dismiss_popups_selenium(driver: WebDriver) -> int:
    """Click known close/accept buttons via hardcoded selectors.

    Returns the number of elements successfully clicked.
    """
    dismissed = 0
    for by, value in _CLOSE_BUTTON_SELECTORS:
        try:
            elements = driver.find_elements(by, value)
            for el in elements:
                if el.is_displayed():
                    try:
                        el.click()
                        dismissed += 1
                        logger.info("Dismissed popup via %s = %s", by, value)
                    except Exception:
                        pass
        except Exception:
            continue
    return dismissed


# ---------------------------------------------------------------------------
# Smart text-matching overlay dismisser
# ---------------------------------------------------------------------------

_DISMISS_WORDS: re.Pattern = re.compile(
    r"^(ok|close|accept|agree|dismiss|allow|got it|continue|skip|"
    r"no thanks|not now|i agree|i accept|allow all|accept all|"
    r"reject all|deny|later|maybe later|remind me later)$",
    re.IGNORECASE,
)


def _is_inside_overlay(driver: WebDriver, el: WebElement) -> bool:
    """Return True if *el* or an ancestor looks like an overlay."""
    try:
        return driver.execute_script("""
            var n = arguments[0];
            while (n && n !== document.body) {
                var c = getComputedStyle(n);
                if (c.position === 'fixed' || c.position === 'sticky') return true;
                if (parseInt(c.zIndex, 10) > 999) return true;
                if (n.getAttribute('role') === 'dialog' ||
                    n.getAttribute('aria-modal') === 'true') return true;
                n = n.parentElement;
            }
            return false;
        """, el)
    except Exception:
        return False


def dismiss_smart_overlay_buttons(driver: WebDriver) -> int:
    """Find visible buttons whose text matches dismiss words AND that
    live inside a modal/overlay, then click them.

    Returns the number of elements clicked.
    """
    dismissed = 0
    try:
        clickables: List[WebElement] = driver.find_elements(
            By.CSS_SELECTOR, "button, a, [role='button']"
        )
    except Exception:
        return 0

    for el in clickables:
        try:
            if not el.is_displayed():
                continue
            text = (el.text or "").strip()
            if not text or len(text) > 40:
                continue
            if not _DISMISS_WORDS.match(text):
                continue
            if not _is_inside_overlay(driver, el):
                continue
            el.click()
            dismissed += 1
            logger.info("Smart-dismissed overlay button: '%s'", text)
        except (StaleElementReferenceException, NoSuchElementException):
            continue
        except Exception:
            continue

    return dismissed


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def handle_popups_light(driver: WebDriver) -> None:
    """Lightweight handler for hot-path hooks (get / click / submit).

    Cost: one native-alert check + one tiny JS round-trip (~1–3 ms).
    No DOM scanning, no selector iteration.
    """
    dismiss_browser_dialogs(driver)
    _ensure_js_dismisser(driver)


def handle_popups(driver: WebDriver) -> None:
    """Full popup handling — **debounced** and **gated**.

    1. Dismiss native browser alerts (cheap, ~0 ms if none).
    2. Ensure the persistent JS dismisser is running (cheap, ~1 ms).
    3. Run fast JS overlay-detection gate (~2–5 ms).
    4. **Only if an overlay is actually visible** → run expensive
       Python-level selector iteration.

    Debounced at 1 second — repeated calls within the window are
    no-ops.
    """
    global _last_full_sweep
    now = time.monotonic()
    if now - _last_full_sweep < _DEBOUNCE_SEC:
        return
    _last_full_sweep = now

    # Cheap: native alert
    dismiss_browser_dialogs(driver)

    # Cheap: ensure JS observer is alive
    _ensure_js_dismisser(driver)

    # Gate: skip expensive Python work if no overlay is visible
    if not _has_visible_overlay(driver):
        return

    logger.debug("Overlay detected — running Python-level popup dismissal")
    dismiss_popups_selenium(driver)
    dismiss_smart_overlay_buttons(driver)
