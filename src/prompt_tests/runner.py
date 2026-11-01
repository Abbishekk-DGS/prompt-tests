"""Load YAML suites, run every (model, case, sample), score, aggregate."""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .providers import Provider
from .scorers import SCORERS, Score


@dataclass
class Case:
    id: str
    input: str
    asserts: list[dict]
    tags: list[str] = field(default_factory=list)
    vars: dict = field(default_factory=dict)


@dataclass
class Suite:
    name: str
    cases: list[Case]
    system: str | None = None
    prompt_template: str = "{input}"
    temperature: float = 0.0
    repeat: int = 1


@dataclass
class CaseResult:
    model: str
    case: Case
    sample: int
    output: str
    latency_s: float
    cached: bool
    scores: list[Score]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(s.passed for s in self.scores)

    @property
    def score(self) -> float:
        return statistics.fmean(s.value for s in self.scores) if self.scores else 0.0


def load_suite(path: Path) -> Suite:
    raw = yaml.safe_load(path.read_text())
    cases = []
    for i, c in enumerate(raw["cases"]):
        asserts = c.get("assert", [])
        if not asserts:
            raise ValueError(f"case {c.get('id', i)} has no assertions")
        for a in asserts:
            kind = next(k for k in a if k in SCORERS) if any(k in SCORERS for k in a) else None
            if kind is None:
                raise ValueError(
                    f"case {c.get('id', i)}: unknown assertion {a}; known: {sorted(SCORERS)}"
                )
        cases.append(
            Case(
                id=str(c.get("id", f"case-{i + 1}")),
                input=c["input"],
                asserts=asserts,
                tags=c.get("tags", []),
                vars=c.get("vars", {}),
            )
        )
    return Suite(
        name=raw.get("name", path.stem),
        cases=cases,
        system=raw.get("system"),
        prompt_template=raw.get("prompt_template", "{input}"),
        temperature=float(raw.get("temperature", 0.0)),
        repeat=int(raw.get("repeat", 1)),
    )


def score_output(output: str, case: Case, judge=None, embed=None) -> list[Score]:
    scores = []
    for a in case.asserts:
        kind = next(k for k in a if k in SCORERS)
        extra = {k: v for k, v in a.items() if k != kind}
        scores.append(
            SCORERS[kind](output, a[kind], case_input=case.input, judge=judge, embed=embed, **extra)
        )
    return scores


def run_suite(
    suite: Suite,
    providers: list[Provider],
    judge: Provider | None = None,
    embed=None,
    workers: int = 2,
    progress=None,
) -> list[CaseResult]:
    jobs = [(p, c, s) for p in providers for c in suite.cases for s in range(suite.repeat)]

    def run(job) -> CaseResult:
        provider, case, sample = job
        prompt = suite.prompt_template.format(input=case.input, **case.vars)
        try:
            kwargs = {"sample": sample} if hasattr(provider, "inner") else {}
            comp = provider.complete(prompt, suite.system, suite.temperature, **kwargs)
            scores = score_output(comp.text, case, judge, embed)
            result = CaseResult(
                provider.name, case, sample, comp.text, comp.latency_s, comp.cached, scores
            )
        except Exception as e:  # a failing model call is a result, not a crash
            result = CaseResult(provider.name, case, sample, "", 0.0, False, [], error=str(e))
        if progress:
            progress(result)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, jobs))


def summarize(results: list[CaseResult]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model in dict.fromkeys(r.model for r in results):
        rs = [r for r in results if r.model == model]
        fresh = sorted(r.latency_s for r in rs if r.error is None)
        tags: dict[str, list[bool]] = {}
        for r in rs:
            for t in r.case.tags or ["untagged"]:
                tags.setdefault(t, []).append(r.passed)
        out[model] = {
            "pass_rate": sum(r.passed for r in rs) / len(rs),
            "mean_score": statistics.fmean(r.score for r in rs),
            "p50_latency": fresh[len(fresh) // 2] if fresh else 0.0,
            "p95_latency": fresh[min(len(fresh) - 1, int(len(fresh) * 0.95))] if fresh else 0.0,
            "errors": sum(r.error is not None for r in rs),
            "n": len(rs),
            "by_tag": {t: sum(v) / len(v) for t, v in sorted(tags.items())},
        }
    return out
