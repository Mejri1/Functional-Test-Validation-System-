"""DOM parser utility.

Provides functions to fetch, clean, and simplify raw HTML so the
Self-Healer agent can reason about page structure without exceeding
LLM context limits.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup, Comment, Tag

logger = logging.getLogger(__name__)

# Tags that are noise for locator analysis
_STRIP_TAGS = {
    "script", "style", "noscript", "svg", "path", "meta", "link",
    "head", "iframe",
}

# Attributes worth keeping for locator inference
_KEEP_ATTRS = {
    "id", "class", "name", "type", "placeholder", "value", "href",
    "aria-label", "aria-labelledby", "aria-describedby", "role",
    "data-testid", "data-test", "data-cy", "data-qa", "for", "title",
    "alt", "src", "action", "method", "autocomplete", "tabindex",
    "data-id", "data-name", "data-value",
}


def clean_html(raw_html: str, max_length: int = 30_000) -> str:
    """Return a simplified version of *raw_html* suitable for LLM analysis.

    Removes scripts, styles, SVGs, comments, and irrelevant attributes
    while preserving the structural and semantic information necessary
    for locator generation.

    Parameters
    ----------
    raw_html : str
        Page source HTML.
    max_length : int, optional
        If the cleaned HTML still exceeds this many characters it will
        be truncated (keeping the first *max_length* chars).
    """
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
    except Exception:
        logger.warning("Failed to parse HTML with html.parser, falling back to raw truncation")
        return raw_html[:max_length]

    # Remove comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove noisy tags entirely
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Strip non-essential attributes
    for tag in soup.find_all(True):
        if not isinstance(tag, Tag):
            continue
        attrs = dict(tag.attrs)
        for attr in list(attrs.keys()):
            if attr not in _KEEP_ATTRS and not attr.startswith("data-"):
                del tag.attrs[attr]

    # Collapse excessive whitespace
    cleaned = soup.prettify()
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "\n<!-- ... truncated ... -->"
        logger.info("Cleaned HTML truncated to %d characters", max_length)

    return cleaned


def extract_interactive_elements(raw_html: str) -> str:
    """Extract ONLY interactive elements (inputs, buttons, links, selects, textareas).

    Returns a simplified text listing that fits small context windows.
    """
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
    except Exception:
        return ""

    lines: list[str] = []
    interactive_tags = ["input", "button", "a", "select", "textarea", "label", "form"]

    for tag in soup.find_all(interactive_tags):
        attrs = {k: v for k, v in tag.attrs.items() if k in _KEEP_ATTRS}
        text = tag.get_text(strip=True)[:80] if tag.string or tag.get_text(strip=True) else ""
        attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        line = f"<{tag.name} {attr_str}>{text}</{tag.name}>" if text else f"<{tag.name} {attr_str}/>"
        lines.append(line)

    return "\n".join(lines)


def get_page_html(driver) -> str:
    """Get current page source from a Selenium WebDriver instance."""
    try:
        return driver.page_source
    except Exception as exc:
        logger.error("Failed to get page source: %s", exc)
        return ""


def get_cleaned_dom(driver, max_length: int = 30_000) -> str:
    """Convenience: get page source from driver and clean it."""
    raw = get_page_html(driver)
    if not raw:
        return ""
    return clean_html(raw, max_length=max_length)
