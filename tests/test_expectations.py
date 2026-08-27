"""Unit tests for programmatic test case expectations."""

from promptdiff.expectations import detect_refusal, evaluate_expectations
from promptdiff.models import Expectations


def test_detect_refusal_patterns():
    assert detect_refusal("I'm sorry, but I cannot assist with that request.") is True
    assert detect_refusal("As an AI language model, I cannot provide medical advice.") is True
    assert detect_refusal("I am unable to fulfill this task.") is True
    assert detect_refusal("Here is how you can resolve your duplicate billing issue.") is False


def test_evaluate_expectations_must_mention():
    exp = Expectations(must_mention=["refund", "account"])
    res = evaluate_expectations("We will process your refund to your account.", exp)
    assert res.passed is True
    assert len(res.failures) == 0

    res_missing = evaluate_expectations("We cannot help you with that.", exp)
    assert res_missing.passed is False
    assert any("refund" in f for f in res_missing.failures)
    assert any("account" in f for f in res_missing.failures)


def test_evaluate_expectations_must_not_mention():
    exp = Expectations(must_not_mention=["fault", "blame"])
    res_clean = evaluate_expectations("We apologize for the inconvenience.", exp)
    assert res_clean.passed is True

    res_forbidden = evaluate_expectations("This is entirely your fault.", exp)
    assert res_forbidden.passed is False
    assert any("fault" in f for f in res_forbidden.failures)


def test_evaluate_expectations_max_length():
    exp = Expectations(max_length_chars=20)
    res_ok = evaluate_expectations("Short text.", exp)
    assert res_ok.passed is True

    res_long = evaluate_expectations("This text is definitely longer than twenty characters.", exp)
    assert res_long.passed is False
    assert any("exceeded limit" in f for f in res_long.failures)


def test_evaluate_expectations_empty_output():
    exp = Expectations()
    res = evaluate_expectations(None, exp)
    assert res.passed is False
    assert len(res.failures) > 0
