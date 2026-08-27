"""LLM-as-judge scoring module for evaluating regressions."""

from __future__ import annotations

import json
import re
from promptdiff.models import JudgeVerdict, TestCase
from promptdiff.providers.base import LLMProvider

JUDGE_SYSTEM_PROMPT = """You are an expert AI quality evaluation judge analyzing changes between two LLM outputs for a regression testing tool called PromptDiff.

You will be given:
1. The System Prompt (the instructions given to the target model)
2. The User Input
3. Expected Properties / Assertions (e.g., must mention keywords, refusal rules, max length)
4. Baseline Output (Output A - the previous accepted output)
5. New Output (Output B - output from the updated prompt/model)

Your task:
Evaluate whether Output B is BETTER, WORSE (regressed), or EQUIVALENT compared to Output A in fulfilling the user query and adhering to instructions and expectations.

Evaluation Rules:
- Do not treat the baseline as automatically correct. Evaluate both outputs independently against the system prompt, user request, and expectations.
- Do not penalize Output B merely because it differs from Output A. A change is a regression only when it causes a meaningful loss in correctness, instruction adherence, usefulness, safety, or required behavior.
- Prefer objective evidence from the provided inputs and expectations over stylistic preference.
- A different phrasing or wording that conveys the exact same meaning is EQUIVALENT.
- If the evidence is insufficient to determine that one output is meaningfully better or worse, choose EQUIVALENT.
- Output B is WORSE if it introduces inaccuracies, violates guidelines/expectations, refuses a legitimate request, drops critical details present in Output A, has severe tone degradation, or exceeds length limits.
- Output B is BETTER if it fixes an error from Output A, improves clarity/conciseness without dropping information, better adheres to guidelines, or handles edge cases more effectively.
- Output B is EQUIVALENT if both outputs are equally valid and neither has a meaningful quality defect over the other.

Respond ONLY with a valid JSON object matching this schema:
{
  "reasoning": "1-2 sentences analyzing key differences and quality impact.",
  "verdict": "better" | "worse" | "equivalent",
  "category": "none" | "meaning_shift" | "tone_shift" | "new_refusal" | "length_violation" | "missing_keyword" | "hallucination" | "improved_clarity" | "improved_adherence",
  "confidence": 0.0 to 1.0
}"""


def _parse_judge_json(raw_text: str) -> JudgeVerdict:
    """Safely extract and parse JSON from judge response."""
    cleaned = raw_text.strip()

    # Remove markdown code blocks if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Attempt direct JSON load
    try:
        data = json.loads(cleaned)
    except Exception:
        # Fallback: extract first JSON-like substring
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception as err:
                return JudgeVerdict(
                    reasoning=f"Failed to parse judge JSON: {err}. Raw: {raw_text[:100]}",
                    verdict="equivalent",
                    category="parsing_error",
                    confidence=0.5,
                )
        else:
            return JudgeVerdict(
                reasoning=f"No JSON object detected in judge response: {raw_text[:100]}",
                verdict="equivalent",
                category="parsing_error",
                confidence=0.5,
            )

    verdict_raw = str(data.get("verdict", "equivalent")).lower().strip()
    if verdict_raw not in {"better", "worse", "equivalent"}:
        verdict_raw = "equivalent"

    return JudgeVerdict(
        reasoning=str(data.get("reasoning", "No explanation provided.")),
        verdict=verdict_raw,
        category=str(data.get("category", "none")),
        confidence=float(data.get("confidence", 1.0)),
    )


class LLMJudge:
    """Evaluates candidate output differences using an LLMProvider."""

    def __init__(self, provider: LLMProvider, judge_model: str | None = None) -> None:
        self.provider = provider
        self.judge_model = judge_model

    def evaluate(
        self,
        case: TestCase,
        baseline_output: str,
        new_output: str,
        system_prompt: str | None = None,
    ) -> JudgeVerdict:
        """Execute the judge prompt against the LLMProvider and return structured verdict."""
        expectations_str = (
            f"- Must Mention: {', '.join(case.expectations.must_mention) if case.expectations.must_mention else 'None'}\n"
            f"- Must Not Mention: {', '.join(case.expectations.must_not_mention) if case.expectations.must_not_mention else 'None'}\n"
            f"- Must Not Refuse: {case.expectations.must_not_refuse}\n"
            f"- Max Length (chars): {case.expectations.max_length_chars or 'None'}"
        )

        user_content = (
            f"[SYSTEM PROMPT]\n{system_prompt or 'None'}\n\n"
            f"[USER INPUT]\n{case.input}\n\n"
            f"[EXPECTATIONS]\n{expectations_str}\n\n"
            f"[BASELINE OUTPUT (Output A)]\n{baseline_output}\n\n"
            f"[NEW OUTPUT (Output B)]\n{new_output}\n"
        )

        result = self.provider.generate(
            user_input=user_content,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            model=self.judge_model,
            temperature=0.0,
            max_tokens=800,
        )

        return _parse_judge_json(result.text)
