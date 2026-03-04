"""Reporter Agent — generates structured JSON and HTML test reports.

Takes all execution results, healing events, page intelligence cache,
and the original Gherkin test to produce comprehensive reports saved to
``outputs/reports/``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage

from llm.factory import get_llm

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "reporter_prompt.txt")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def _load_system_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _screenshot_to_base64(path: str) -> str:
    """Read a screenshot file and return a base64-encoded data URI string.

    Returns an empty string if the file does not exist or cannot be read.
    """
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{data}"
    except Exception as exc:
        logger.debug("Could not encode screenshot %s: %s", path, exc)
        return ""


def _embed_screenshots_in_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """Walk *report_data* and replace screenshot file paths with base64 URIs.

    This makes the HTML report fully self-contained (no external image files).
    """
    for step in report_data.get("steps_detail", []):
        path = step.get("screenshot_path") or step.get("screenshot") or ""
        if path:
            b64 = _screenshot_to_base64(path)
            step["screenshot_base64"] = b64
        else:
            step["screenshot_base64"] = ""
    return report_data


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response."""
    import re

    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    patterns = [
        r"```json\s*\n?(.*?)```",
        r"```\s*\n?(.*?)```",
        r"\{.*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                return json.loads(candidate.strip())
            except (json.JSONDecodeError, IndexError):
                continue

    raise ValueError(f"Could not extract valid JSON from reporter response:\n{text[:500]}")


def _generate_report_data_fallback(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate report data without LLM if the LLM call fails."""
    execution_results = state.get("execution_results", {})
    healing_history = state.get("healing_history", [])
    gherkin = state.get("gherkin", "")
    url = state.get("url", "")

    steps = execution_results.get("steps", [])
    passed = sum(1 for s in steps if s.get("status") == "PASSED")
    failed = sum(1 for s in steps if s.get("status") == "FAILED")
    healed = sum(1 for h in healing_history if h.get("healed"))

    if failed == 0:
        overall = "PASSED"
    elif passed == 0:
        overall = "FAILED"
    else:
        overall = "PARTIALLY_PASSED"

    if execution_results.get("overall_status") == "ERROR":
        overall = "FAILED"

    # ── Build page intelligence report entries ─────────────────────
    pi_cache = execution_results.get("page_intelligence_cache", {})
    # Also merge the top-level page_intelligence from the old Explorer
    top_pi = state.get("page_intelligence", {})
    if top_pi and url and url not in pi_cache:
        pi_cache[url] = top_pi

    page_intel_entries = []
    for pg_url, pg_intel in pi_cache.items():
        entry = dict(pg_intel)
        entry["url"] = pg_url
        # Screenshot thumbnail (base64 embedded)
        ss_path = entry.get("screenshot_path", "")
        entry["screenshot_base64"] = _screenshot_to_base64(ss_path)
        page_intel_entries.append(entry)

    return {
        "report_title": "Test Validation Report",
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall,
        "target_url": url,
        "gherkin_test": gherkin,
        "page_intelligence": state.get("page_intelligence", {}),
        "page_intel_entries": page_intel_entries,
        "summary": {
            "total_steps": len(steps),
            "passed_steps": passed,
            "failed_steps": failed,
            "healed_steps": healed,
            "total_execution_time_seconds": 0,
            "healing_attempts": len(healing_history),
            "successful_healings": healed,
        },
        "steps_detail": [
            {
                "step_number": s.get("step", i + 1),
                "action": s.get("action", ""),
                "status": s.get("status", "UNKNOWN"),
                "error": s.get("error"),
                "screenshot_path": s.get("screenshot"),
                "healing_applied": False,
                "engine": s.get("engine", "selenium"),
                "notes": "",
                "intel_summary": s.get("intel_summary", ""),
            }
            for i, s in enumerate(steps)
        ],
        "healing_events": [
            {
                "step_number": h.get("step"),
                "original_locator": h.get("original_locator", ""),
                "new_locator": f"{h.get('new_locator_type', '')}: {h.get('new_locator_value', '')}",
                "success": h.get("healed", False),
                "attempts": h.get("attempts", 0),
                "reasoning": h.get("analysis", ""),
            }
            for h in healing_history
        ],
        "unresolved_failures": [
            {
                "step_number": s.get("step", i + 1),
                "description": s.get("error", "Unknown failure"),
                "suggested_fix": "Review the test step and element locators manually.",
            }
            for i, s in enumerate(steps)
            if s.get("status") == "FAILED"
        ],
        "recommendations": [],
    }


def run_reporter(state: Dict[str, Any]) -> Dict[str, Any]:
    """Reporter agent node for the LangGraph workflow.

    Reads execution results, healing history, and original Gherkin from
    state.  Produces ``final_report_paths`` with JSON and HTML report paths.
    """
    logger.info("Reporter agent starting — generating test report")

    gherkin = state.get("gherkin", "")
    url = state.get("url", "")
    execution_results = state.get("execution_results", {})
    healing_history = state.get("healing_history", [])
    retry_count = state.get("retry_count", 0)
    page_intelligence = state.get("page_intelligence", {})

    # Build context for LLM
    pi_section = ""
    if page_intelligence:
        pi_section = (
            f"## Page Intelligence\n"
            f"```json\n{json.dumps(page_intelligence, indent=2, default=str)[:2000]}\n```\n\n"
        )

    # Include per-page intelligence from execution
    pi_cache = execution_results.get("page_intelligence_cache", {})
    if pi_cache:
        pi_section += (
            f"## Per-Page Intelligence ({len(pi_cache)} pages)\n"
            f"```json\n{json.dumps(list(pi_cache.keys()), indent=2)}\n```\n\n"
        )

    user_message = (
        f"## Original Gherkin Test\n```\n{gherkin}\n```\n\n"
        f"## Target URL\n{url}\n\n"
        f"{pi_section}"
        f"## Execution Results\n```json\n{json.dumps(execution_results, indent=2, default=str)[:6000]}\n```\n\n"
        f"## Self-Healing History\n```json\n{json.dumps(healing_history, indent=2, default=str)[:3000]}\n```\n\n"
        f"## Retry Count: {retry_count}\n\n"
        f"Generate the JSON report now."
    )

    # Try LLM-based report generation
    report_data = None
    try:
        system_prompt = _load_system_prompt()
        llm = get_llm(temperature=0.1, max_tokens=4096)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        report_data = _extract_json(response.content)
        logger.info("Reporter LLM produced report data successfully")
    except Exception as exc:
        logger.warning("Reporter LLM failed (%s), using fallback report generation", exc)
        report_data = None

    # Fall back to programmatic report if LLM failed
    if not report_data:
        report_data = _generate_report_data_fallback(state)

    # Ensure required fields
    report_data.setdefault("timestamp", datetime.now().isoformat())
    report_data.setdefault("overall_status", "UNKNOWN")
    report_data.setdefault("target_url", url)
    report_data.setdefault("gherkin_test", gherkin)

    # ── Always build page_intel_entries for the HTML template ──────
    if "page_intel_entries" not in report_data:
        pi_cache_raw = execution_results.get("page_intelligence_cache", {})
        top_pi = page_intelligence
        if top_pi and url and url not in pi_cache_raw:
            pi_cache_raw[url] = top_pi
        entries = []
        for pg_url, pg_intel in pi_cache_raw.items():
            entry = dict(pg_intel)
            entry["url"] = pg_url
            entry["screenshot_base64"] = _screenshot_to_base64(entry.get("screenshot_path", ""))
            entries.append(entry)
        report_data["page_intel_entries"] = entries

    # Ensure steps have intel_summary
    for sd in report_data.get("steps_detail", []):
        sd.setdefault("intel_summary", "")

    # Save JSON report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(REPORT_DIR, f"report_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    logger.info("JSON report saved to %s", json_path)

    # Generate HTML report
    html_path = os.path.join(REPORT_DIR, f"report_{ts}.html")
    try:
        # Embed screenshots as base64 data URIs for self-contained HTML
        _embed_screenshots_in_report(report_data)

        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
        template = env.get_template("report_template.html")
        html_content = template.render(report=report_data)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info("HTML report saved to %s", html_path)
    except Exception as exc:
        logger.error("HTML report generation failed: %s", exc)
        # Generate a minimal HTML fallback
        html_content = _minimal_html_report(report_data)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return {
        "final_report_paths": {
            "json": json_path,
            "html": html_path,
        }
    }


def _minimal_html_report(data: Dict[str, Any]) -> str:
    """Generate a minimal HTML report if Jinja2 template fails."""
    _embed_screenshots_in_report(data)
    status = data.get("overall_status", "UNKNOWN")
    color = {"PASSED": "#22c55e", "FAILED": "#ef4444", "PARTIALLY_PASSED": "#f59e0b"}.get(status, "#64748b")
    steps_html = ""
    for step in data.get("steps_detail", []):
        s_status = step.get("status", "UNKNOWN")
        s_color = {"PASSED": "#22c55e", "FAILED": "#ef4444"}.get(s_status, "#64748b")
        engine = step.get("engine", "selenium")
        engine_badge = (
            '<span style="display:inline-block;font-size:0.65rem;font-weight:700;'
            'padding:2px 8px;border-radius:8px;margin-left:8px;'
            f'{"background:rgba(59,130,246,0.15);color:#3b82f6" if engine == "selenium" else "background:rgba(139,92,246,0.15);color:#8b5cf6"}'
            f'">{engine.replace("_", " ").title()}</span>'
        )
        ss_html = ""
        b64 = step.get("screenshot_base64", "")
        if b64:
            ss_html = f'<br><img src="{b64}" style="max-width:100%;margin-top:8px;border-radius:6px;" />'
        steps_html += (
            f"<tr><td>{step.get('step_number', '')}</td>"
            f"<td>{step.get('action', '')}{engine_badge}</td>"
            f"<td style='color:{s_color};font-weight:bold'>{s_status}</td>"
            f"<td>{step.get('error', '') or ''}{ss_html}</td></tr>"
        )
    return f"""<!DOCTYPE html>
<html><head><title>Test Report</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;}}
table{{border-collapse:collapse;width:100%;}}
th,td{{padding:8px 12px;border:1px solid #ddd;text-align:left;}}
th{{background:#f5f5f5;}}</style></head>
<body><h1>Test Validation Report</h1>
<p>Status: <span style="color:{color};font-weight:bold;font-size:1.2em">{status}</span></p>
<p>URL: {data.get('target_url','')}</p>
<p>Time: {data.get('timestamp','')}</p>
<h2>Steps</h2>
<table><tr><th>Step</th><th>Action</th><th>Status</th><th>Error</th></tr>
{steps_html}</table></body></html>"""
