"""Popup / overlay handler utility.

Injects JavaScript into the browser to automatically dismiss common
overlays, cookie banners, GDPR notices, newsletter popups, etc.
Also provides a Python-level dismisser that clicks known close buttons
and a smart catch-all that finds overlay buttons by visible text.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

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

# ---------------------------------------------------------------------------
# JavaScript auto-dismisser
# ---------------------------------------------------------------------------

POPUP_DISMISSER_JS = r"""
(function() {
    'use strict';

    // Common selectors for cookie / GDPR / overlay close buttons
    const CLOSE_SELECTORS = [
        // Cookie consent
        '[id*="cookie"] button', '[class*="cookie"] button',
        '[id*="consent"] button', '[class*="consent"] button',
        '[id*="gdpr"] button', '[class*="gdpr"] button',
        '[aria-label*="cookie" i]', '[aria-label*="consent" i]',
        '[aria-label*="close" i]', '[aria-label*="dismiss" i]',
        '[aria-label*="accept" i]',
        // Generic close / dismiss
        'button[class*="close"]', 'button[class*="dismiss"]',
        'button[class*="accept"]', 'a[class*="close"]',
        '.modal .close', '.overlay .close',
        '[data-dismiss="modal"]', '[data-action="close"]',
        // Common cookie-consent libraries
        '#onetrust-accept-btn-handler',
        '.cc-btn.cc-dismiss', '.cc-btn.cc-allow',
        '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
        '.cky-btn-accept', '#accept-cookies',
        '#cookie-accept', '.cookie-accept',
        'button[data-cookiefirst-action="accept"]',
        '.js-cookie-consent-agree',
        // Newsletter popups
        '.popup-close', '.newsletter-close',
        '[class*="popup"] [class*="close"]',
        '[class*="newsletter"] button[class*="close"]',
        '.modal-close', '.modal__close',
    ];

    // ---------- smart catch-all (JS side) ----------
    const DISMISS_WORDS = /^(ok|close|accept|agree|dismiss|allow|got it|continue|skip|no thanks|not now|i agree|i accept|allow all|accept all|reject all|deny|later|maybe later|remind me later)$/i;

    function isInsideOverlay(el) {
        var node = el;
        while (node && node !== document.body) {
            try {
                var cs = window.getComputedStyle(node);
                if (cs.position === 'fixed' || cs.position === 'sticky') return true;
                var z = parseInt(cs.zIndex, 10);
                if (z > 999) return true;
                if (node.getAttribute('role') === 'dialog' ||
                    node.getAttribute('aria-modal') === 'true') return true;
            } catch(e) {}
            node = node.parentElement;
        }
        return false;
    }

    function smartDismiss() {
        var clickables = document.querySelectorAll('button, a, [role="button"]');
        for (var i = 0; i < clickables.length; i++) {
            try {
                var el = clickables[i];
                if (el.offsetParent === null) continue;
                var txt = (el.innerText || el.textContent || '').trim();
                if (txt.length > 40) continue;
                if (DISMISS_WORDS.test(txt) && isInsideOverlay(el)) {
                    el.click();
                }
            } catch(e) {}
        }
    }

    function clickMatching(selectors) {
        for (const sel of selectors) {
            try {
                const els = document.querySelectorAll(sel);
                for (const el of els) {
                    if (el.offsetParent !== null) {
                        el.click();
                    }
                }
            } catch(e) {}
        }
    }

    // Remove common overlay / backdrop elements
    function removeOverlays() {
        const overlaySelectors = [
            '.modal-backdrop', '.overlay-backdrop',
            '[class*="overlay"]', '[class*="backdrop"]',
            '[id*="overlay"]', '[id*="backdrop"]',
        ];
        for (const sel of overlaySelectors) {
            try {
                const els = document.querySelectorAll(sel);
                for (const el of els) {
                    if (el.children.length === 0 || el.classList.contains('backdrop')) {
                        el.style.display = 'none';
                    }
                }
            } catch(e) {}
        }
    }

    function sweep() {
        clickMatching(CLOSE_SELECTORS);
        removeOverlays();
        smartDismiss();
    }

    // Run immediately + delayed sweeps
    sweep();
    setTimeout(sweep, 1500);
    setTimeout(sweep, 3000);
    setTimeout(sweep, 5000);

    // MutationObserver for late popups
    var observer = new MutationObserver(function() { sweep(); });
    if (document.body) {
        observer.observe(document.body, { childList: true, subtree: true });
    }
    setTimeout(function() { observer.disconnect(); }, 30000);
})();
"""


def inject_popup_dismisser(driver: WebDriver) -> None:
    """Inject the JavaScript popup dismisser into the current page."""
    try:
        driver.execute_script(POPUP_DISMISSER_JS)
        logger.debug("Popup dismisser JS injected")
    except Exception as exc:
        logger.warning("Failed to inject popup dismisser JS: %s", exc)


# ---------------------------------------------------------------------------
# Native browser dialog handler
# ---------------------------------------------------------------------------

def dismiss_browser_dialogs(driver: WebDriver) -> int:
    """Accept any pending native Chrome alert / confirm / prompt dialogs.

    Returns the number of alerts dismissed.
    """
    dismissed = 0
    # Try up to 3 times in case stacked alerts exist
    for _ in range(3):
        try:
            alert = driver.switch_to.alert
            alert_text = alert.text
            alert.accept()
            dismissed += 1
            logger.info("Dismissed native alert: %s", alert_text[:120])
        except NoAlertPresentException:
            break
        except WebDriverException:
            break
    return dismissed


# ---------------------------------------------------------------------------
# Python-level overlay dismisser (hardcoded selectors)
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
    """Click common close/accept buttons via hardcoded selectors.

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
                        time.sleep(0.25)
                    except Exception:
                        pass
        except Exception:
            continue
    return dismissed


# ---------------------------------------------------------------------------
# Smart catch-all: find overlay buttons by visible text (Python side)
# ---------------------------------------------------------------------------

_DISMISS_WORDS: re.Pattern = re.compile(
    r"^(ok|close|accept|agree|dismiss|allow|got it|continue|skip|"
    r"no thanks|not now|i agree|i accept|allow all|accept all|"
    r"reject all|deny|later|maybe later|remind me later)$",
    re.IGNORECASE,
)


def _is_inside_overlay(driver: WebDriver, el: WebElement) -> bool:
    """Return True if *el* or one of its ancestors looks like an overlay.

    Checks for ``position: fixed/sticky``, ``z-index > 999``,
    ``role=dialog``, or ``aria-modal=true``.
    """
    try:
        return driver.execute_script("""
            var node = arguments[0];
            while (node && node !== document.body) {
                var cs = window.getComputedStyle(node);
                if (cs.position === 'fixed' || cs.position === 'sticky') return true;
                var z = parseInt(cs.zIndex, 10);
                if (z > 999) return true;
                if (node.getAttribute('role') === 'dialog' ||
                    node.getAttribute('aria-modal') === 'true') return true;
                node = node.parentElement;
            }
            return false;
        """, el)
    except Exception:
        return False


def dismiss_smart_overlay_buttons(driver: WebDriver) -> int:
    """Find ALL visible buttons/links whose text matches common dismiss
    words AND that live inside a modal/overlay, then click them.

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
            time.sleep(0.25)
        except (StaleElementReferenceException, NoSuchElementException):
            continue
        except Exception:
            continue

    return dismissed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def handle_popups(driver: WebDriver) -> None:
    """Full popup handling:

    1. Dismiss any native browser alert/confirm/prompt dialogs.
    2. Inject the JavaScript auto-dismisser (hardcoded selectors +
       smart text-matching inside overlays).
    3. Click known close buttons via Selenium (hardcoded selectors).
    4. Smart catch-all: find visible overlay buttons by text and click.
    """
    dismiss_browser_dialogs(driver)
    inject_popup_dismisser(driver)
    time.sleep(0.5)
    dismiss_popups_selenium(driver)
    dismiss_smart_overlay_buttons(driver)
