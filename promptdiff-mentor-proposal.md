# PromptDiff — LLM Observability & Regression-Testing Tool
### Project Proposal for Mentor Review

---

## One-Line Description
A CLI tool that catches AI output regressions before they ship — diffs LLM responses across prompt/model changes using embedding-drift detection and LLM-as-judge scoring, with GitHub Actions integration.

---

## 1. Problem Statement

Companies building AI-powered features (chatbots, summarizers, support assistants) frequently change the underlying model or prompt — upgrading to a newer model, tweaking instructions, adjusting parameters. Unlike traditional software, there's no compiler error or failing unit test when an AI feature gets *worse*. Teams typically find out only after users complain or a business metric drops.

Traditional software has CI/CD pipelines with automated regression tests. AI features almost never have an equivalent — most teams eyeball a few outputs manually and ship. This is a known, widely-discussed gap in the AI engineering ecosystem (the same problem tools like PromptFoo and Braintrust are trying to solve commercially).

## 2. What the Project Does

1. Developer defines a set of test cases (input + expected properties — not exact text, but qualities like "should mention X," "shouldn't refuse," "should stay under Y length").
2. Run the test suite against the current prompt/model → saved as a baseline.
3. Developer makes a change (new prompt, new model, new settings).
4. Run the test suite again → PromptDiff compares old vs. new outputs automatically.
5. Report shows exactly which test cases improved, which regressed, and *why* (meaning shift, tone shift, length change, new refusals).
6. Optionally wired into GitHub Actions — runs automatically on every pull request and can block a merge if outputs clearly got worse.

**In short:** a spell-checker for AI quality — it doesn't catch typos, it catches "the AI got dumber," automatically, every time something changes.

## 3. Core Features

| Feature | Description | Priority |
|---|---|---|
| YAML/JSON test case format | Define input + expected properties per test | Must-have |
| CLI test runner | Runs all test cases against a given model/prompt config | Must-have |
| Output diffing | Compares old vs. new output per test case | Must-have |
| Embedding-drift check | Fast/cheap "how different is the meaning" score | Must-have |
| LLM-as-judge scoring | Deeper check on borderline/flagged cases — "better, worse, or equivalent, and why" | Should-have |
| Run history (SQLite) | Stores past runs so trends are visible over time | Should-have |
| Web dashboard | Simple visual trend view of pass/fail over time | Should-have |
| GitHub Action integration | Auto-runs on PRs, comments a diff summary, can fail the check | Nice-to-have (high demo impact) |

## 4. Technical Architecture

```
[test_cases.yaml] 
      → CLI (Python, Typer)
      → For each test case: call Claude API, store output
      → Compare against baseline run:
          1. Embedding similarity check (fast first pass)
          2. LLM-as-judge (Claude API) for flagged/borderline cases only
      → Results saved to SQLite (run history)
      → Report printed to terminal + optional React dashboard
      → GitHub Action wraps the CLI, posts PR comment, 
         fails check on regression past threshold
```

**Stack:**
- CLI: Python + Typer
- AI layer: Claude API (both for generating test outputs and for judge comparisons)
- Embeddings: lightweight embedding model for similarity scoring
- Storage: SQLite (no external infra needed)
- Dashboard (optional): React + TypeScript, served via a small FastAPI layer
- CI: GitHub Actions

**Design decision worth discussing with mentor:** the pipeline is split into 3 distinct AI-related steps (generate → embedding-check → judge-check) rather than one big prompt, specifically to keep costs down (judge calls only run on flagged cases) and make failure points easier to debug.

## 5. Test Data / Validation Plan

Since there's no "real" production AI feature to test against, the plan is to:
- Build a small example AI feature (e.g., a support-reply generator or summarizer) as the test subject
- Write 15–20 test cases covering typical inputs
- Deliberately introduce a few "bad" prompt changes to prove the tool catches real regressions live in a demo

## 6. Build Timeline (Solo, ~4 Weeks)

| Week | Focus |
|---|---|
| 1 | CLI skeleton, test case format, basic run + output storage |
| 2 | Embedding-drift diffing + LLM-as-judge scoring, threshold tuning |
| 3 | SQLite run history + React dashboard trend view |
| 4 | GitHub Action integration, polish, demo prep, README |

## 7. Why This Project

- Addresses a real, currently underserved gap in the AI engineering ecosystem (LLM eval/observability tooling)
- Matches current hiring demand — most AI-shipping teams need this and few have it built properly
- Fully solo-buildable without needing external data access, compliance approval, or expensive infrastructure
- Demos well: a clear "before/after — caught the regression" moment is fast and visually convincing

## 8. Open Questions for Mentor

- Is the LLM-as-judge approach (using Claude to grade Claude's own outputs) methodologically sound enough, or is there a better way to validate judge reliability?
- Should the scope include multi-model comparison (e.g., testing prompt behavior across Claude vs. GPT vs. Gemini), or is single-model regression testing enough for a strong MVP?
- Any recommended datasets/benchmarks to validate the embedding-drift scoring approach against, beyond self-written test cases?
- Feedback on whether the GitHub Action integration is worth prioritizing over the dashboard, given time constraints before placements.
