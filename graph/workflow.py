"""LangGraph workflow definition.

Builds the stateful graph that orchestrates all agents:

  START → explorer → analyst → script_writer → executor
                                                  │
                                         ┌────────┤
                                         │        │
                                   has_failures?  all_passed?
                                         │        │
                                         ▼        ▼
                                     healer    reporter → END
                                         │
                                    re-executor
                                         │
                                  max_retries? → reporter → END
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from graph.state import GraphState
from agents.explorer import run_explorer
from agents.analyst import run_analyst
from agents.script_writer import run_script_writer
from agents.executor import run_executor
from agents.self_healer import run_self_healer
from agents.reporter import run_reporter

logger = logging.getLogger(__name__)

MAX_HEAL_RETRIES = 3


# ── Node wrappers (add logging / progress) ─────────────────────────

def explorer_node(state: GraphState) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("  AGENT: Explorer — Gathering page intelligence")
    logger.info("=" * 60)
    print("\n🔎 [1/6] Explorer Agent — visiting page and gathering intelligence...")
    result = run_explorer(state)
    pi = result.get("page_intelligence", {})
    print(f"   ✓ Page type: {pi.get('page_type', '?')}")
    return result


def analyst_node(state: GraphState) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("  AGENT: Analyst — Parsing Gherkin test case")
    logger.info("=" * 60)
    print("\n🔍 [2/6] Analyst Agent — parsing Gherkin into action plan...")
    result = run_analyst(state)
    plan = result.get("action_plan", [])
    print(f"   ✓ Generated {len(plan)} actions")
    for a in plan:
        print(f"     Step {a.get('step')}: {a.get('action')} → {a.get('target')}")
    return result


def script_writer_node(state: GraphState) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("  AGENT: Script Writer — Generating Selenium script")
    logger.info("=" * 60)
    print("\n📝 [3/6] Script Writer Agent — generating Selenium test script...")
    result = run_script_writer(state)
    script = result.get("selenium_script", "")
    print(f"   ✓ Generated script ({len(script)} characters)")
    return result


def executor_node(state: GraphState) -> Dict[str, Any]:
    retry_count = state.get("retry_count", 0)
    label = f" (retry {retry_count})" if retry_count > 0 else ""
    logger.info("=" * 60)
    logger.info("  AGENT: Executor — Running test%s", label)
    logger.info("=" * 60)
    print(f"\n🚀 [4/6] Executor Agent — running test in Chrome{label}...")
    result = run_executor(state)
    exec_results = result.get("execution_results", {})
    status = exec_results.get("overall_status", "UNKNOWN")
    steps = exec_results.get("steps", [])
    passed = sum(1 for s in steps if s.get("status") == "PASSED")
    failed = sum(1 for s in steps if s.get("status") == "FAILED")
    print(f"   Result: {status} — {passed} passed, {failed} failed out of {len(steps)} steps")
    failed_list = result.get("failed_steps", [])
    if failed_list:
        print(f"   ⚠ {len(failed_list)} steps flagged for self-healing")
    return result


def healer_node(state: GraphState) -> Dict[str, Any]:
    retry_count = state.get("retry_count", 0)
    logger.info("=" * 60)
    logger.info("  AGENT: Self-Healer — Attempt %d/%d", retry_count + 1, MAX_HEAL_RETRIES)
    logger.info("=" * 60)
    print(f"\n🔧 [5/6] Self-Healer Agent — analyzing DOM and fixing locators "
          f"(attempt {retry_count + 1}/{MAX_HEAL_RETRIES})...")
    result = run_self_healer(state)
    history = result.get("healing_history", [])
    healed = sum(1 for h in history if h.get("healed"))
    print(f"   ✓ Healing complete — {healed} locators healed so far")
    return result


def reporter_node(state: GraphState) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("  AGENT: Reporter — Generating reports")
    logger.info("=" * 60)
    print("\n📊 [6/6] Reporter Agent — generating test reports...")
    result = run_reporter(state)
    paths = result.get("final_report_paths", {})
    if paths:
        print(f"   ✓ JSON report: {paths.get('json', 'N/A')}")
        print(f"   ✓ HTML report: {paths.get('html', 'N/A')}")
    return result


# ── Conditional routing ─────────────────────────────────────────────

def should_heal_or_report(state: GraphState) -> str:
    """After execution, decide whether to heal or go straight to report."""
    failed_steps = state.get("failed_steps", [])
    retry_count = state.get("retry_count", 0)

    if failed_steps and retry_count < MAX_HEAL_RETRIES:
        logger.info("Routing to healer — %d failed steps, retry %d/%d",
                     len(failed_steps), retry_count, MAX_HEAL_RETRIES)
        return "healer"
    else:
        if retry_count >= MAX_HEAL_RETRIES:
            logger.info("Routing to reporter — max retries reached")
        else:
            logger.info("Routing to reporter — no healable failures")
        return "reporter"


def after_healing(state: GraphState) -> str:
    """After healing, decide whether to re-execute or give up."""
    failed_steps = state.get("failed_steps", [])
    retry_count = state.get("retry_count", 0)

    if failed_steps and retry_count < MAX_HEAL_RETRIES:
        logger.info("Routing back to executor for retry %d", retry_count)
        return "executor"
    else:
        logger.info("Routing to reporter after healing (retries=%d, remaining_failures=%d)",
                     retry_count, len(failed_steps))
        return "reporter"


# ── Build the graph ─────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph workflow."""

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("explorer", explorer_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("script_writer", script_writer_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("healer", healer_node)
    workflow.add_node("reporter", reporter_node)

    # Set entry point
    workflow.set_entry_point("explorer")

    # Linear edges
    workflow.add_edge("explorer", "analyst")
    workflow.add_edge("analyst", "script_writer")
    workflow.add_edge("script_writer", "executor")

    # Conditional: after execution → heal or report
    workflow.add_conditional_edges(
        "executor",
        should_heal_or_report,
        {
            "healer": "healer",
            "reporter": "reporter",
        },
    )

    # Conditional: after healing → re-execute or report
    workflow.add_conditional_edges(
        "healer",
        after_healing,
        {
            "executor": "executor",
            "reporter": "reporter",
        },
    )

    # Reporter → END
    workflow.add_edge("reporter", END)

    return workflow.compile()


def run_workflow(
    gherkin: str,
    url: str,
    headless: bool = True,
) -> Dict[str, Any]:
    """Convenience function: build graph, inject initial state, and run.

    Returns the final state dict.
    """
    graph = build_graph()

    initial_state: GraphState = {
        "gherkin": gherkin,
        "url": url,
        "action_plan": [],
        "selenium_script": "",
        "execution_results": {},
        "failed_steps": [],
        "healing_history": [],
        "retry_count": 0,
        "final_report_paths": {},
        "error": "",
        "headless": headless,
        "page_intelligence": {},
    }

    logger.info("Starting workflow for URL: %s", url)
    print("\n" + "=" * 60)
    print("  AUTOMATED TEST VALIDATION SYSTEM")
    print("=" * 60)
    print(f"  URL: {url}")
    print(f"  Headless: {headless}")
    print(f"  Gherkin steps: {gherkin.count(chr(10)) + 1} lines")
    print("=" * 60)

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("  WORKFLOW COMPLETE")
    print("=" * 60)

    return final_state
