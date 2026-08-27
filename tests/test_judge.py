"""Unit tests for LLM judge evaluation and JSON parsing."""

from promptdiff.judge import LLMJudge, _parse_judge_json
from promptdiff.models import TestCase
from promptdiff.providers.base import GenerationResult


class MockJudgeProvider:
    def __init__(self, return_text: str):
        self.return_text = return_text

    def generate(self, user_input: str, **kwargs) -> GenerationResult:
        return GenerationResult(text=self.return_text)


def test_parse_judge_json_clean():
    json_str = '{"reasoning": "Output B is more concise and addresses the query.", "verdict": "better", "category": "improved_clarity", "confidence": 0.95}'
    verdict = _parse_judge_json(json_str)
    assert verdict.verdict == "better"
    assert verdict.category == "improved_clarity"
    assert verdict.confidence == 0.95
    assert "more concise" in verdict.reasoning


def test_parse_judge_json_markdown_wrapped():
    wrapped = """```json
{
  "reasoning": "Output B refused the request when it should not have.",
  "verdict": "worse",
  "category": "new_refusal",
  "confidence": 0.9
}
```"""
    verdict = _parse_judge_json(wrapped)
    assert verdict.verdict == "worse"
    assert verdict.category == "new_refusal"


def test_parse_judge_json_fallback_on_invalid():
    invalid_str = "This is not valid json at all."
    verdict = _parse_judge_json(invalid_str)
    assert verdict.verdict == "equivalent"
    assert verdict.category == "parsing_error"


def test_llm_judge_execution():
    mock_resp = '{"reasoning": "Tone shifted negatively.", "verdict": "worse", "category": "tone_shift", "confidence": 0.85}'
    provider = MockJudgeProvider(return_text=mock_resp)
    judge = LLMJudge(provider=provider)

    case = TestCase(id="tc1", input="Can you refund me?")
    verdict = judge.evaluate(
        case=case,
        baseline_output="Sure, I can process your refund.",
        new_output="No refunds allowed ever.",
    )

    assert verdict.verdict == "worse"
    assert verdict.category == "tone_shift"
    assert verdict.confidence == 0.85
