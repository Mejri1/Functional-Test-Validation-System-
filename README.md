# Automated Web Test Validation System

A multi-agent AI system that automatically validates web applications by parsing Gherkin test cases, generating Selenium scripts, executing them in a real browser, and self-healing broken locators — all without human intervention.

## Architecture

```
START → Analyst Agent → Script Writer Agent → Executor Agent
                                                    │
                                          ┌─────────┤
                                          │         │
                                     failures?   all passed?
                                          │         │
                                          ▼         ▼
                                    Self-Healer   Reporter Agent → END
                                       Agent
                                          │
                                     re-execute
                                          │
                                   max retries? → Reporter Agent → END
```

### Agents

| Agent | Role |
|-------|------|
| **Analyst** | Parses Gherkin test cases into structured JSON action plans |
| **Script Writer** | Generates complete, runnable Selenium Python scripts with robust locator strategies |
| **Executor** | Runs the script in a real Chrome browser, captures results, screenshots, and errors |
| **Self-Healer** | Analyzes the live DOM when locators fail, generates alternative locators, patches the script |
| **Reporter** | Produces detailed JSON and HTML reports with step-by-step results and healing events |

## Tech Stack

- **LLM**: Multi-provider — Groq, LM Studio (local), or Cerebras (switchable via `.env`)
- **Agent Framework**: LangGraph + LangChain
- **Browser Automation**: Selenium 4 + ChromeDriver (via webdriver-manager)
- **DOM Parsing**: BeautifulSoup4
- **Report Generation**: Jinja2 HTML templates + JSON
- **Config**: python-dotenv

## Setup

### 1. Prerequisites

- Python 3.10+
- Google Chrome browser installed
- An LLM API key — one of:
  - Groq (free at [console.groq.com/keys](https://console.groq.com/keys))
  - Cerebras ([cerebras.ai](https://cerebras.ai/))
  - LM Studio running locally (no key needed)

### 2. Install Dependencies

```bash
cd test-validation-system
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env — set LLM_PROVIDER and the matching API key
# See "Switching LLM Provider" below for details
```

### 4. Verify Installation

```bash
python -m pytest tests/test_sample.py -v
```

## Usage

### From a .feature file

```bash
python main.py --gherkin tests/sample.feature --url "https://example.com"
```

### Inline Gherkin

```bash
python main.py --gherkin "Given I am on the homepage\nWhen I look at the page\nThen I should see the heading 'Example Domain'" --url "https://example.com"
```

### With Visible Browser (for debugging)

```bash
python main.py --gherkin tests/sample.feature --url "https://example.com" --no-headless
```

## Example Run

```bash
python main.py \
  --gherkin "Given I am on the Example Domain homepage\nWhen I look at the page content\nThen I should see the heading 'Example Domain'\nAnd I should see a 'More information...' link" \
  --url "https://example.com"
```

**Expected output:**
```
============================================================
  AUTOMATED TEST VALIDATION SYSTEM
============================================================
  URL: https://example.com
  Headless: True
  Gherkin steps: 4 lines
============================================================

🔍 [1/5] Analyst Agent — parsing Gherkin into action plan...
   ✓ Generated 6 actions

📝 [2/5] Script Writer Agent — generating Selenium test script...
   ✓ Generated script (3245 characters)

🚀 [3/5] Executor Agent — running test in Chrome...
   Result: PASSED — 6 passed, 0 failed out of 6 steps

📊 [5/5] Reporter Agent — generating test reports...
   ✓ JSON report: outputs/reports/report_20260228_143022.json
   ✓ HTML report: outputs/reports/report_20260228_143022.html

============================================================
  FINAL RESULT: ✅  PASSED
============================================================
```

## Project Structure

```
test-validation-system/
├── agents/                   # LLM-powered agent implementations
│   ├── analyst.py            # Gherkin → action plan
│   ├── script_writer.py      # Action plan → Selenium script
│   ├── executor.py           # Runs script in Chrome
│   ├── self_healer.py        # DOM analysis & locator healing
│   └── reporter.py           # JSON + HTML report generation
├── llm/                      # LLM provider abstraction
│   └── factory.py            # Universal get_llm() factory
├── graph/                    # LangGraph workflow
│   ├── state.py              # Shared TypedDict state
│   └── workflow.py           # Graph definition & routing
├── tools/                    # Utility modules
│   ├── selenium_runner.py    # Chrome setup & script execution
│   ├── dom_parser.py         # HTML cleaning & parsing
│   └── popup_handler.py      # Popup/overlay auto-dismisser
├── prompts/                  # LLM system prompts
│   ├── analyst_prompt.txt
│   ├── script_writer_prompt.txt
│   ├── healer_prompt.txt
│   └── reporter_prompt.txt
├── templates/
│   └── report_template.html  # Jinja2 HTML report template (dark theme, visual step flow)
├── outputs/
│   ├── reports/              # Generated JSON & HTML reports
│   ├── screenshots/
│   │   ├── success/          # Screenshots of passing steps
│   │   └── failure/          # Screenshots of failing steps
│   └── generated_scripts/    # Saved Selenium scripts
├── tests/
│   ├── sample.feature        # Example Gherkin test
│   └── test_sample.py        # Unit tests for components
├── main.py                   # CLI entry point
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Outputs

After each run, the system generates:

- **JSON Report**: `outputs/reports/report_{timestamp}.json` — machine-readable results
- **HTML Report**: `outputs/reports/report_{timestamp}.html` — visual report with base64-embedded screenshots per step (self-contained, open in any browser)
- **Screenshots**: Captured on **every** step (pass and fail):
  - `outputs/screenshots/success/` — screenshots of successful steps
  - `outputs/screenshots/failure/` — screenshots of failed steps
- **Scripts**: `outputs/generated_scripts/` — the generated Selenium scripts for inspection

## Self-Healing

When a Selenium locator fails (element not found, timeout, etc.):

1. The Executor flags the failed step
2. The Self-Healer agent receives the failed step description + live page DOM
3. The LLM analyzes the DOM and generates 3 alternative locators ranked by confidence
4. The best locator is patched into the script
5. The Executor re-runs the test (up to 3 retries per failed locator)

## Switching LLM Provider

Edit your `.env` file to swap providers with **zero code changes**.

### Groq (cloud, default)

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_your_key_here
```

Other Groq models: `meta-llama/llama-4-scout-17b-16e-instruct`, `mixtral-8x7b-32768`

### LM Studio (local)

Start LM Studio, load a model, then:

```env
LLM_PROVIDER=lmstudio
LLM_MODEL=meta-llama-3.1-8b-instruct
# No API key needed — runs on http://127.0.0.1:1234
```

Override the server address if needed: `LMSTUDIO_BASE_URL=http://192.168.1.100:1234/v1`

### Cerebras (cloud)

```env
LLM_PROVIDER=cerebras
LLM_MODEL=llama-3.3-70b
CEREBRAS_API_KEY=your_cerebras_key_here
```

Other Cerebras models: `llama-3.1-8b`, `llama-3.1-70b`

## Known Limitations

1. **Dynamic SPAs**: Heavily JavaScript-rendered pages may need longer wait times. The system uses explicit waits but very complex SPAs might still pose challenges.
2. **CAPTCHA**: The system cannot solve CAPTCHAs. Sites with CAPTCHA on critical paths will fail.
3. **2FA/MFA**: Multi-factor authentication flows require manual credentials and may not be fully automatable.
4. **File Downloads**: File download verification is not currently supported.
5. **iframes**: Cross-origin iframe interactions have limited support.
6. **Rate Limits**: Groq/Cerebras API free tiers have rate limits. If you hit them, add a delay between runs or use a paid plan. LM Studio has no rate limits.
7. **Locator Healing**: Self-healing works best when the target element exists on the page but has a different selector. Completely missing elements cannot be healed.
8. **Chrome Version**: webdriver-manager auto-detects Chrome version, but very new Chrome versions may briefly lack matching ChromeDriver.

## License

MIT
