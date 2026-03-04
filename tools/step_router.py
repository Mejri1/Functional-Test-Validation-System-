"""Step Router — decides which execution engine handles each step.

Routes steps tagged with a ``complexity`` field by the Analyst agent:

* ``standard`` → Selenium (handles normal actions **and** file upload/download)
* ``captcha`` / ``two_factor`` → Browser Use (vision-capable LLM required)

File uploads and downloads are **always** handled by Selenium via
``send_keys()`` on ``<input type="file">`` — no Browser Use needed.
"""

from __future__ import annotations

from typing import Dict

# Complexity values that require the Browser Use engine.
# NOTE: file_upload is intentionally NOT here — Selenium handles it.
_BROWSER_USE_COMPLEXITIES = frozenset({"captcha", "two_factor"})


def route_step(step: Dict) -> str:
    """Return ``"browser_use"`` or ``"selenium"`` based on step complexity.

    Parameters
    ----------
    step : dict
        A single action dict from the Analyst's action plan.
        Expected to contain a ``"complexity"`` key.

    Returns
    -------
    str
        ``"browser_use"`` if the step requires advanced handling,
        ``"selenium"`` otherwise.
    """
    complexity = step.get("complexity", "standard")
    if complexity in _BROWSER_USE_COMPLEXITIES:
        return "browser_use"
    return "selenium"
