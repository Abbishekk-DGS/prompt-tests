"""Scorers: each takes the model output (+ the case) and returns a score in [0, 1] and a reason."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Score:
    name: str
    value: float
    passed: bool
    reason: str = ""


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().strip(".").split())


def exact(output: str, value: str, **_) -> Score:
    ok = _norm(output) == _norm(str(value))
    return Score("exact", float(ok), ok, "" if ok else f"expected {value!r}")


def contains(output: str, value, **_) -> Score:
    needles = value if isinstance(value, list) else [value]
    found = [n for n in needles if _norm(str(n)) in _norm(output)]
    frac = len(found) / len(needles)
    missing = [n for n in needles if n not in found]
    return Score("contains", frac, frac == 1.0, f"missing {missing}" if missing else "")


def not_contains(output: str, value, **_) -> Score:
    needles = value if isinstance(value, list) else [value]
    present = [n for n in needles if _norm(str(n)) in _norm(output)]
    return Score(
        "not_contains", float(not present), not present, f"found {present}" if present else ""
    )


def regex(output: str, value: str, **_) -> Score:
    ok = re.search(value, output, re.S | re.I) is not None
    return Score("regex", float(ok), ok, "" if ok else f"no match for /{value}/")


def numeric(output: str, value: float, tolerance: float = 1e-6, **_) -> Score:
    """Last number in the output (where answers usually end up) within tolerance."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?", output)
    if not nums:
        return Score("numeric", 0.0, False, "no number in output")
    got = float(nums[-1].replace(",", ""))
    ok = abs(got - float(value)) <= tolerance * max(1.0, abs(float(value)))
    return Score("numeric", float(ok), ok, "" if ok else f"got {got}, expected {value}")


def json_schema(output: str, value: dict, **_) -> Score:
    import jsonschema

    try:
        data = json.loads(_strip_fences(output))
    except json.JSONDecodeError as e:
        return Score("json_schema", 0.0, False, f"invalid JSON: {e.msg}")
    try:
        jsonschema.validate(data, value)
    except jsonschema.ValidationError as e:
        return Score("json_schema", 0.0, False, f"schema: {e.message}")
    return Score("json_schema", 1.0, True)


def max_words(output: str, value: int, **_) -> Score:
    n = len(output.split())
    return Score("max_words", float(n <= value), n <= value, f"{n} words" if n > value else "")


def semantic(
    output: str, value: str, threshold: float = 0.75, embed: Callable | None = None, **_
) -> Score:
    """Cosine similarity between output and reference embeddings."""
    if embed is None:
        return Score("semantic", 0.0, False, "no embedding model configured")
    a, b = embed([output, str(value)])
    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return Score("semantic", sim, sim >= threshold, f"similarity {sim:.2f}")


JUDGE_PROMPT = """You are a strict evaluator. Grade the RESPONSE against the RUBRIC.

QUESTION:
{input}

RESPONSE:
{output}

RUBRIC:
{rubric}

Think step by step about each rubric point, then end with exactly one line:
SCORE: <integer 1-5>
(5 = fully meets every rubric point, 1 = fails)"""


def llm_judge(
    output: str, value: str, case_input: str = "", judge=None, threshold: int = 4, **_
) -> Score:
    """LLM-as-judge with a rubric and a parsed 1-5 score (G-Eval style)."""
    if judge is None:
        return Score("llm_judge", 0.0, False, "no judge model configured")
    reply = judge.complete(JUDGE_PROMPT.format(input=case_input, output=output, rubric=value)).text
    m = re.findall(r"SCORE:\s*([1-5])", reply)
    if not m:
        return Score("llm_judge", 0.0, False, "judge gave no score")
    s = int(m[-1])
    rationale = [ln.strip() for ln in reply.split("SCORE:")[0].splitlines() if ln.strip()]
    reason = rationale[-1][:160] if rationale else "no rationale given"
    return Score("llm_judge", (s - 1) / 4, s >= threshold, f"{s}/5 · {reason}")


SCORERS: dict[str, Callable[..., Score]] = {
    "exact": exact,
    "contains": contains,
    "not_contains": not_contains,
    "regex": regex,
    "numeric": numeric,
    "json_schema": json_schema,
    "max_words": max_words,
    "semantic": semantic,
    "llm_judge": llm_judge,
}
