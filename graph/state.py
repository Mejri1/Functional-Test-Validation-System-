"""Shared state definition for the LangGraph workflow.

This module defines the TypedDict used as the state schema for the
multi-agent test validation graph.  Every agent node reads from and
writes to this state object.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class GraphState(TypedDict, total=False):
    """State that flows through the LangGraph workflow.

    Attributes
    ----------
    gherkin : str
        Raw Gherkin test case text (Given/When/Then).
    url : str
        Target website URL to test against.
    page_intelligence : dict
        Page characteristics detected by the Explorer agent (login form,
        captcha, 2FA, file upload, anti-bot measures, etc.).
    page_intel_cache : dict
        Per-URL page intelligence gathered during execution by the
        seamless Explorer integration in selenium_runner.  Keyed by
        URL string, each value is the full intel dict for that page.
    action_plan : list[dict]
        Structured list of UI actions produced by the Analyst agent.
    selenium_script : str
        Complete runnable Selenium Python script produced by the Script
        Writer agent.
    execution_results : dict
        Results dictionary returned by the Executor agent, containing
        per-step pass/fail status, error messages, screenshots, etc.
    failed_steps : list[dict]
        Steps that failed during execution and may need self-healing.
    healing_history : list[dict]
        Log of all self-healing attempts (what broke, what was tried,
        whether it succeeded).
    retry_count : int
        Current number of self-healing retry iterations.
    final_report_paths : dict
        Paths to the generated JSON and HTML report files.
    error : str
        Top-level error message if the workflow crashes.
    headless : bool
        Whether Chrome should run in headless mode.
    """

    gherkin: str
    url: str
    page_intelligence: Dict[str, Any]
    page_intel_cache: Dict[str, Dict[str, Any]]
    action_plan: List[Dict[str, Any]]
    selenium_script: str
    execution_results: Dict[str, Any]
    failed_steps: List[Dict[str, Any]]
    healing_history: List[Dict[str, Any]]
    retry_count: int
    final_report_paths: Dict[str, str]
    error: str
    headless: bool
