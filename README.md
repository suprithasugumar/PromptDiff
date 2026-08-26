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
- Google Gemini API Key (Free Tier via [Google AI Studio](https://aistudio.google.com/))
- *(Optional)* Anthropic API Key (if evaluating Claude models)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suprithasugumar/PromptDiff.git
   cd PromptDiff
   ```

2. **Set up virtual environment & install dependencies:**
   Using `uv`:
   ```bash
   uv sync
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
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
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set your `GEMINI_API_KEY`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## Usage

### Quick Start (Dry Run)
```bash
promptdiff run examples/support_bot/test_cases.yaml --dry-run
```

### Run Test Suite
```bash
promptdiff run examples/support_bot/test_cases.yaml
```

The target provider (`gemini` or `anthropic`), model name, and system prompt are configured directly in your test suite YAML file (see [examples/support_bot/test_cases.yaml](examples/support_bot/test_cases.yaml)).

---

## License

This project is licensed under the [MIT License](LICENSE).
