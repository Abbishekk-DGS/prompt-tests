from pathlib import Path

import numpy as np
import pytest

from prompt_tests.providers import CachedProvider, Completion, make_provider
from prompt_tests.report import render
from prompt_tests.runner import load_suite, run_suite, summarize
from prompt_tests.scorers import (
    contains,
    exact,
    json_schema,
    llm_judge,
    max_words,
    not_contains,
    numeric,
    regex,
    semantic,
)

SUITE = Path(__file__).parents[1] / "suites" / "core.yaml"


class FakeModel:
    """Answers from a lookup table; counts calls so caching can be verified."""

    def __init__(self, name, answers):
        self.name, self.answers, self.calls = name, answers, 0

    def complete(self, prompt, system=None, temperature=0.0):
        self.calls += 1
        for key, answer in self.answers.items():
            if key in prompt:
                return Completion(answer, 0.5)
        return Completion("I don't know", 0.5)


def test_scorers():
    assert exact("Canberra.", "canberra").passed
    assert contains("Red, green and blue", ["red", "blue"]).passed
    assert contains("red", ["red", "blue"]).value == 0.5
    assert not not_contains("it is hot", ["hot"]).passed
    assert regex("Yes, because...", r"^\W*yes").passed
    assert numeric("The answer is 1,391.0", 1391).passed
    assert numeric("17*23 = 391", 391).passed and not numeric("391? no, 390", 391).passed
    assert max_words("one two three", 3).passed and not max_words("a b c d", 3).passed


def test_json_schema_scorer_handles_fences_and_errors():
    schema = {"type": "object", "required": ["age"], "properties": {"age": {"type": "integer"}}}
    assert json_schema('```json\n{"age": 3}\n```', schema).passed
    assert "schema" in json_schema('{"age": "3"}', schema).reason
    assert "invalid JSON" in json_schema("nope", schema).reason


def test_semantic_and_judge_with_fakes():
    embed = lambda texts: np.array([[1.0, 0.0], [0.8, 0.6]])  # noqa: E731  cos = 0.8
    assert semantic("a", "b", threshold=0.75, embed=embed).passed
    assert not semantic("a", "b", threshold=0.9, embed=embed).passed

    judge = FakeModel("judge", {"RUBRIC": "Clear and correct.\nSCORE: 4"})
    s = llm_judge("answer", "be correct", judge=judge)
    assert s.passed and s.value == 0.75 and s.reason.startswith("4/5")


def test_suite_loads_and_validates():
    suite = load_suite(SUITE)
    assert len(suite.cases) == 12 and suite.temperature == 0
    assert {t for c in suite.cases for t in c.tags} >= {"reasoning", "structured", "judged"}


def test_unknown_assertion_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("cases:\n  - input: hi\n    assert:\n      - telepathy: yes\n")
    with pytest.raises(ValueError, match="unknown assertion"):
        load_suite(bad)


def test_run_compare_report_and_cache(tmp_path):
    suite = load_suite(SUITE)
    good = FakeModel("good", {"17 * 23": "391", "capital of Australia": "Canberra"})
    bad = FakeModel("bad", {"17 * 23": "392"})
    cached_good = CachedProvider(good, tmp_path / "c.sqlite")
    results = run_suite(suite, [cached_good, bad], workers=1)
    summary = summarize(results)
    assert summary["good"]["by_tag"]["reasoning"] > summary["bad"]["by_tag"]["reasoning"]

    calls = good.calls
    run_suite(suite, [cached_good], workers=1)
    assert good.calls == calls  # everything came from the cache

    page = render(suite.name, results)
    assert page.startswith("<!doctype html>") and "core-skills" in page and "Canberra" in page


def test_provider_specs():
    assert make_provider("ollama:qwen2.5:3b").name == "ollama:qwen2.5:3b"
    p = make_provider("openai:llama@http://localhost:8080/v1")
    assert p.base_url == "http://localhost:8080/v1"
    with pytest.raises(ValueError):
        make_provider("cloud:gpt")
