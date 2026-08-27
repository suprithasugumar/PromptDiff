# PromptDiff

> **Catch AI regressions before they ship.** Automated regression testing, semantic drift detection, and observability for LLM prompts and models.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Actions CI](https://img.shields.io/badge/CI-GitHub_Actions-green.svg)](.github/workflows/promptdiff.yml)

---

## 💡 Why PromptDiff?

Upgrading an LLM model version (e.g., GPT-4o, Gemini 2.5 Flash, Claude 3.5 Sonnet), tuning temperature, or refactoring system prompts can cause subtle regressions:
- **Meaning drift:** Answers hallucinate new facts or change recommendations.
- **Tone shift:** Helpful responses become curt or robotic.
- **Assertion failure:** The model omits required legal disclaimers or refuses benign user inputs.
- **Length ballooning:** Token usage spikes unexpectedly.

Unlike traditional software with compiler errors and unit tests, AI features silently degrade. **PromptDiff brings CI/CD rigor to LLM engineering**:
1. Run your test suite against your current configuration to establish a **ground-truth baseline**.
2. Make your prompt or model changes.
3. Re-run PromptDiff: outputs are diffed using a fast **3-tier evaluation pipeline**:
   - **Tier 1 — Programmatic Assertions:** Instant regex/keyword checks (`must_mention`, `must_not_mention`, `must_not_refuse`, `max_length_chars`).
   - **Tier 2 — Embedding Drift Detection:** Cosine similarity scoring (`gemini-embedding-001`) to detect semantic shifts.
   - **Tier 3 — LLM-as-Judge:** Automated deep evaluation on borderline or regressed cases only (saving API cost and latency).
4. View results in the terminal, SQLite historical database, interactive Web Dashboard, or as automated GitHub Pull Request comments.

```
[test_cases.yaml] 
      │
      ▼
 ┌───────────────┐
 │ CLI Test      │ ──► Call Target Model (Gemini / Claude)
 │ Runner        │
 └───────┬───────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3-Tier Diff Engine                                      │
 │                                                        │
 │ 1. Assertions (must_mention, must_not_refuse, lengths) │
 │ 2. Embedding Drift (Cosine Similarity < 0.88)          │
 │ 3. LLM-as-Judge (Evaluates "better / worse / equiv")    │
 └───────┬────────────────────────────────────────────────┘
         │
         ├──► 🖥️ Terminal Rich Report
         ├──► 🗄️ SQLite History (`promptdiff.db`)
         ├──► 🌐 Interactive Web Dashboard (`promptdiff dashboard`)
         └──► 🤖 GitHub Action PR Comments & CI Check Failure
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- Google Gemini API Key (Free Tier available via [Google AI Studio](https://aistudio.google.com/))
- *(Optional)* Anthropic API Key (if testing Claude models)

### Installation

#### Using `uv` (Fastest):
```bash
git clone https://github.com/suprithasugumar/PromptDiff.git
cd PromptDiff

# Install virtual environment and dependencies
uv sync
```

#### Using standard `pip`:
```bash
git clone https://github.com/suprithasugumar/PromptDiff.git
cd PromptDiff

python -m venv .venv
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -e .
```

### Environment Configuration

Copy the example environment file and add your Google Gemini API key:
```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=AIzaSy...your_gemini_api_key
# Optional:
ANTHROPIC_API_KEY=sk-ant-...
```

> **How to get a Gemini API Key:**
> 1. Visit [Google AI Studio](https://aistudio.google.com/).
> 2. Sign in and click **"Get API key"**.
> 3. Create a free API key and paste it into `.env` (or set as environment variable `export GEMINI_API_KEY="your-key"`).

---

## 📝 Writing Test Cases

Test suites are written in human-readable YAML. Each suite specifies the target model configuration and test scenarios with qualitative and quantitative expectations.

Example ([`examples/support_bot/test_cases.yaml`](examples/support_bot/test_cases.yaml)):

```yaml
version: "1"
name: "customer-support-eval"
description: "Evaluation suite for customer service assistant"

target:
  provider: "gemini"               # "gemini" or "anthropic"
  model: "gemini-2.5-flash"        # Target model ID
  temperature: 0.2                 # Sampling temperature
  max_tokens: 500
  system_prompt: |
    You are a friendly, helpful customer support agent for AcmeCorp.
    Always apologize politely for issues, explain return windows (30 days),
    and guide customers step-by-step.

test_cases:
  - id: "return_policy_inquiry"
    description: "Customer asking about return windows and procedure"
    input: "How do I return a damaged item I bought 2 weeks ago?"
    expectations:
      must_mention:
        - "30 days"
        - "refund"
      must_not_mention:
        - "cannot help"
      must_not_refuse: true
      max_length_chars: 600

  - id: "angry_customer_refund"
    description: "Handle angry customer demanding immediate refund"
    input: "My order #9921 never arrived! This is ridiculous! Refund me right now!"
    expectations:
      must_mention:
        - "apologize"
        - "refund"
      must_not_refuse: true
      max_length_chars: 800
```

### Supported Expectations
- `must_mention`: List of substrings/keywords that **must** appear in the output (case-insensitive).
- `must_not_mention`: List of forbidden substrings that **must not** appear.
- `must_not_refuse`: Boolean assertion verifying the model did not refuse benign requests (e.g. "I cannot fulfill this request").
- `max_length_chars`: Maximum character length upper bound.

---

## 💻 Local CLI Usage

### 1. Dry Run (No API Calls)
Validate your test suite YAML syntax without consuming API tokens:
```bash
promptdiff run examples/support_bot/test_cases.yaml --dry-run
```

### 2. Establish a Baseline
Execute the suite against the current model and record it as the golden baseline:
```bash
promptdiff run examples/support_bot/test_cases.yaml --baseline
```
*Baselines are serialized to `runs/<suite>_baseline.json` and recorded in SQLite.*

### 3. Diff Against Baseline
Modify your system prompt, model, or parameters in the YAML file and run:
```bash
promptdiff run examples/support_bot/test_cases.yaml
```
PromptDiff will:
1. Generate new responses.
2. Compute cosine similarity against baseline embeddings.
3. Trigger the LLM judge on any case where assertions failed or similarity dropped below the threshold.
4. Render a Rich terminal summary and detailed diagnostic breakdown for flagged cases.

### 4. Adjusting the Embedding Drift Threshold
```bash
# Set custom similarity threshold (default: 0.88)
promptdiff run examples/support_bot/test_cases.yaml --threshold 0.85
```

> **📌 Note on the 0.88 Embedding Threshold:**  
> The default similarity threshold of `0.88` (using `gemini-embedding-001`) is an empirical baseline heuristic. In practice, minor rephrasing and synonym changes typically score `0.90–0.98`, while significant behavioral or factual shifts drop below `0.85`. Cases scoring below `0.88` trigger Tier 3 LLM-as-judge inspection. You can tune `--threshold` to match your application's domain variance.

### 5. CI Severity & Failure Policies
Configure when PromptDiff returns a non-zero exit code (blocking CI):
```bash
# Fail on any regression or error (default)
promptdiff run examples/support_bot/test_cases.yaml --fail-on any

# Fail only if hard expectations (must_mention, refusal, length) failed
promptdiff run examples/support_bot/test_cases.yaml --fail-on hard

# Fail if LLM judge marked output as "worse" or hard expectations failed
promptdiff run examples/support_bot/test_cases.yaml --fail-on judge-worse

# Advisory mode (always exits with code 0)
promptdiff run examples/support_bot/test_cases.yaml --fail-on none

# Allow up to N regressions before failing
promptdiff run examples/support_bot/test_cases.yaml --max-regressions 1
```

### 6. Query Run History
Inspect regression trends across all runs stored in SQLite:
```bash
# View recent runs across all suites
promptdiff history

# Filter history by suite name
promptdiff history customer-support-eval

# Inspect granular test case results for a specific run ID
promptdiff history --run-id run-20260827-142500-a1b2c3d4
```

### 7. Interactive Web Dashboard
Launch the built-in web dashboard to visualize trends, pass rates, and output diffs:
```bash
promptdiff dashboard
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🤖 GitHub Action CI Integration

PromptDiff integrates directly into GitHub Actions to test pull requests and post sticky diff comments.

### 1. Add Repository Secret
In your GitHub repository, go to **Settings > Secrets and variables > Actions** and add:
- `GEMINI_API_KEY`: Your Google AI Studio Gemini API key.

### 2. Add Workflow (`.github/workflows/promptdiff.yml`)
```yaml
name: PromptDiff Regression Check

on:
  pull_request:
    branches: [main, master]
    paths:
      - 'examples/**'
      - 'prompts/**'
      - 'src/**'

permissions:
  contents: read
  pull-requests: write

jobs:
  eval:
    name: LLM Regression Test
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install uv & Python
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install 3.11

      - name: Install PromptDiff
        run: uv sync --frozen

      - name: Run PromptDiff Suite
        id: promptdiff
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: |
          uv run promptdiff run examples/support_bot/test_cases.yaml \
            --markdown-report pr_comment.md \
            --fail-on any
        continue-on-error: true

      - name: Post or Update PR Comment
        uses: actions/github-script@v7
        if: always() && github.event_name == 'pull_request'
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const fs = require('fs');
            const commentFile = 'pr_comment.md';
            if (!fs.existsSync(commentFile)) return;
            const body = fs.readFileSync(commentFile, 'utf8');
            const commentTag = '<!-- promptdiff-pr-comment -->';
            const fullBody = `${commentTag}\n${body}`;

            const { data: comments } = await github.rest.issues.listComments({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
            });

            const existingComment = comments.find(c => c.body && c.body.includes(commentTag));

            if (existingComment) {
              await github.rest.issues.updateComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                comment_id: existingComment.id,
                body: fullBody,
              });
            } else {
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body: fullBody,
              });
            }

      - name: Enforce Check Status
        if: steps.promptdiff.outcome == 'failure'
        run: |
          echo "PromptDiff detected regressions above configured threshold."
          exit 1
```

---

## 🧪 Testing & Validation

Run the automated test suite locally:
```bash
uv run pytest -v
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
