#!/usr/bin/env python3
"""CLI entry point for the Automated Test Validation System.

Usage
-----
# From a .feature file:
python main.py --gherkin path/to/test.feature --url https://example.com

# Inline Gherkin:
python main.py --gherkin "Given I am on the homepage\\nWhen I click login\\nThen I see the login form" --url https://example.com

# With visible browser for debugging:
python main.py --gherkin test.feature --url https://example.com --no-headless
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ── Ensure project root is on sys.path ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Load environment variables ──────────────────────────────────────
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Logging setup ───────────────────────────────────────────────────
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# StreamHandler shows INFO+, file shows DEBUG+
logging.getLogger().handlers[0].setLevel(logging.DEBUG)
logging.getLogger().handlers[1].setLevel(logging.INFO)

# Quieten noisy third-party loggers
for noisy in ("urllib3", "selenium", "httpx", "httpcore", "openai", "groq"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("main")


def _read_gherkin(source: str) -> str:
    """Read Gherkin text from a file path or treat as inline text."""
    path = Path(source)
    if path.exists() and path.is_file():
        logger.info("Reading Gherkin from file: %s", path)
        return path.read_text(encoding="utf-8").strip()

    # Treat as inline text — unescape literal \\n sequences
    text = source.replace("\\n", "\n").strip()
    if not text:
        raise ValueError("Empty Gherkin input")
    return text


def _validate_env() -> None:
    """Ensure required environment variables are set."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.error("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        print("\n❌  ERROR: GROQ_API_KEY environment variable is not set.")
        print("   1. Copy .env.example to .env")
        print("   2. Add your Groq API key to the .env file")
        print("   3. Get a free key at https://console.groq.com/keys\n")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Web Test Validation System — "
                    "run Gherkin tests against real websites using AI agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py --gherkin tests/sample.feature --url https://example.com\n'
            '  python main.py --gherkin "Given I am on the homepage" --url https://example.com\n'
            '  python main.py --gherkin tests/sample.feature --url https://example.com --no-headless\n'
        ),
    )
    parser.add_argument(
        "--gherkin", "-g",
        required=True,
        help="Path to a .feature file OR inline Gherkin text (use \\n for newlines).",
    )
    parser.add_argument(
        "--url", "-u",
        required=True,
        help="Target website URL to test against.",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        default=False,
        help="Run Chrome in visible (headed) mode for debugging.",
    )

    args = parser.parse_args()

    # Validate
    _validate_env()

    gherkin_text = _read_gherkin(args.gherkin)
    url = args.url.strip()
    headless = not args.no_headless

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    logger.info("Gherkin input:\n%s", gherkin_text)
    logger.info("Target URL: %s", url)
    logger.info("Headless: %s", headless)

    # ── Run the workflow ────────────────────────────────────────────
    from graph.workflow import run_workflow

    try:
        final_state = run_workflow(gherkin=gherkin_text, url=url, headless=headless)
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Workflow crashed: %s", exc, exc_info=True)
        print(f"\n❌  Workflow error: {exc}")
        sys.exit(1)

    # ── Print final summary ─────────────────────────────────────────
    report_paths = final_state.get("final_report_paths", {})
    exec_results = final_state.get("execution_results", {})
    healing_history = final_state.get("healing_history", [])

    overall = exec_results.get("overall_status", "UNKNOWN")
    status_icon = {"PASSED": "✅", "FAILED": "❌", "PARTIALLY_PASSED": "⚠️"}.get(overall, "❓")

    print("\n" + "=" * 60)
    print(f"  FINAL RESULT: {status_icon}  {overall}")
    print("=" * 60)

    steps = exec_results.get("steps", [])
    if steps:
        passed = sum(1 for s in steps if s.get("status") == "PASSED")
        failed = sum(1 for s in steps if s.get("status") == "FAILED")
        print(f"  Steps: {passed} passed, {failed} failed, {len(steps)} total")

    if healing_history:
        healed = sum(1 for h in healing_history if h.get("healed"))
        print(f"  Self-healing: {healed}/{len(healing_history)} locators healed")

    if report_paths:
        print(f"\n  📄 JSON Report: {report_paths.get('json', 'N/A')}")
        print(f"  🌐 HTML Report: {report_paths.get('html', 'N/A')}")

    print(f"  📝 Debug Log:   {log_file}")
    print()


if __name__ == "__main__":
    main()
