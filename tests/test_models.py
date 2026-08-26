"""Unit tests for models and YAML loading."""

import pytest
from promptdiff.models import TestSuite, TestCase, Expectations, TargetConfig


def test_load_support_bot_yaml():
    yaml_path = "examples/support_bot/test_cases.yaml"
    suite = TestSuite.from_yaml(yaml_path)
    assert suite.name == "support-reply-generator"
    assert suite.target.provider == "gemini"
    assert suite.target.model == "gemini-3.6-flash"
    assert len(suite.test_cases) == 4
    assert suite.test_cases[0].id == "billing_double_charge"
    assert "refund" in suite.test_cases[0].expectations.must_mention


def test_duplicate_test_case_id_rejected():
    with pytest.raises(ValueError, match="Duplicate test case id"):
        TestSuite(
            name="duplicate_test",
            test_cases=[
                TestCase(id="tc1", input="hello"),
                TestCase(id="tc1", input="world"),
            ],
        )
