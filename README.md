# PromptDiff

> Catch AI output regressions before they ship — automated testing for LLM prompts and models.

---

## Overview

**PromptDiff** is a developer-focused CLI tool designed to catch regressions in AI applications across prompt and model iterations. Similar to traditional regression suites in software engineering, PromptDiff enables teams to define test cases with qualitative expectations and quantitatively monitor outputs over time.

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Anthropic API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suprithasugumar/PromptDiff.git
   cd PromptDiff
   ```

2. **Set up virtual environment & install dependencies:**
   Using `uv`:
   ```bash
   uv venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate

   uv pip install -e .
   ```

   Using standard `pip`:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate

   pip install -e .
   ```

3. **Configure Environment:**
   Copy the example environment file and set your Anthropic API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and provide your `ANTHROPIC_API_KEY`.

---

## Usage

<!-- Usage documentation placeholder — will be populated in upcoming releases -->
*Usage instructions and CLI reference will be documented here.*

### Quick Start (Dry Run)
```bash
promptdiff run examples/support_bot/test_cases.yaml --dry-run
```

---

## License

This project is licensed under the [MIT License](LICENSE).
