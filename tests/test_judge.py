import pytest
import json
from judge.pipeline import parse_structured_verdict, LLMJudge
from judge.bias import run_mitigated_pairwise
from judge.validation import run_adversarial_probe

def test_parse_structured_verdict_markdown():
    raw = "```json\n{\n  \"score\": 5,\n  \"rationale\": \"Perfect answer.\"\n}\n```"
    parsed = parse_structured_verdict(raw, expected_type="pointwise")
    assert parsed["score"] == 5
    assert parsed["rationale"] == "Perfect answer."

def test_parse_structured_verdict_malformed():
    raw = "Here is my response:\n{\n  \"score\": 4,\n  \"rationale\": \"Good but incomplete.\"\n}\nHope this helps!"
    parsed = parse_structured_verdict(raw, expected_type="pointwise")
    assert parsed["score"] == 4
    assert "Good but incomplete" in parsed["rationale"]

def test_parse_structured_verdict_pairwise_fallback():
    raw = "The winner is A because of correctness."
    parsed = parse_structured_verdict(raw, expected_type="pairwise")
    assert parsed["winner"] == "A"

def test_mock_judge_pointwise():
    judge = LLMJudge(judge_provider="mock")
    verdict = judge.evaluate_pointwise(
        query="What is the replication factor?",
        context="The replication factor is 3.",
        expected_output="3",
        model_output="The replication factor is 3.",
        criterion="correctness"
    )
    assert verdict["score"] in [1, 2, 3, 4, 5]
    assert "rationale" in verdict

def test_mock_judge_pairwise_mitigated():
    judge = LLMJudge(judge_provider="mock")
    # A has correct fact, B has "wrong" which triggers mock B loss or vice versa
    res = run_mitigated_pairwise(
        judge,
        query="What is the replication factor?",
        context="The replication factor is 3.",
        expected_output="3",
        response_a="The replication factor is 3.",
        response_b="The replication factor is 5, which is wrong."
    )
    # The consensus should be computed
    assert "consensus_winner" in res
    assert res["consensus_winner"] in ["A", "B", "TIE"]
