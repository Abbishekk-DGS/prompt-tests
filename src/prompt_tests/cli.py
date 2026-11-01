"""prompt-tests: test your prompts and models like code.

prompt-tests run suites/core.yaml -m ollama:qwen2.5:1.5b -m ollama:qwen2.5:3b
prompt-tests run suites/core.yaml -m ollama:qwen2.5:3b --judge ollama:qwen2.5:3b --min-pass 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx
import numpy as np

from . import __version__
from .providers import OLLAMA_URL, CachedProvider, make_provider
from .report import render
from .runner import load_suite, run_suite, summarize


def ollama_embedder(model: str):
    def embed(texts: list[str]) -> np.ndarray:
        r = httpx.post(
            f"{OLLAMA_URL}/api/embed", json={"model": model, "input": texts}, timeout=120
        )
        r.raise_for_status()
        return np.array(r.json()["embeddings"])

    return embed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prompt-tests",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} · by Abbishekk DGS · abbishekkdgs.tech",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run a suite against one or more models")
    r.add_argument("suite")
    r.add_argument(
        "-m",
        "--model",
        action="append",
        required=True,
        help="ollama:<model> or openai:<model>@<url>",
    )
    r.add_argument("--judge", help="model for llm_judge assertions")
    r.add_argument(
        "--embed", default="nomic-embed-text", help="Ollama embedding model for `semantic`"
    )
    r.add_argument("--out", default="prompt-tests-report")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--no-cache", action="store_true")
    r.add_argument(
        "--min-pass", type=float, default=None, help="exit 1 if any model's pass rate is below this"
    )
    args = parser.parse_args(argv)

    suite = load_suite(Path(args.suite))
    cache = Path(".prompt-tests/cache.sqlite")
    providers = [make_provider(m) for m in args.model]
    if not args.no_cache:
        providers = [CachedProvider(p, cache) for p in providers]
    judge = make_provider(args.judge) if args.judge else None
    if judge and not args.no_cache:
        judge = CachedProvider(judge, cache)

    total = len(providers) * len(suite.cases) * suite.repeat
    done = 0

    def progress(res) -> None:
        nonlocal done
        done += 1
        mark = "✓" if res.passed else "✗"
        print(f"[{done}/{total}] {mark} {res.model:<28} {res.case.id}")

    results = run_suite(
        suite, providers, judge, ollama_embedder(args.embed), args.workers, progress
    )
    summary = summarize(results)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(render(suite.name, results))
    (out / "results.json").write_text(
        json.dumps(
            {
                "suite": suite.name,
                "summary": summary,
                "results": [
                    {
                        "model": r.model,
                        "case": r.case.id,
                        "sample": r.sample,
                        "passed": r.passed,
                        "score": r.score,
                        "latency_s": r.latency_s,
                        "output": r.output,
                        "error": r.error,
                        "scores": [asdict(s) for s in r.scores],
                    }
                    for r in results
                ],
            },
            indent=1,
        )
    )
    print()
    width = max(len(m) for m in summary)
    for m, s in summary.items():
        print(
            f"{m:<{width}}  pass {s['pass_rate']:6.1%}  score {s['mean_score']:.2f}  p50 {s['p50_latency']:.1f}s"
        )
    print(f"\nreport -> {out / 'report.html'}")

    if args.min_pass is not None and any(s["pass_rate"] < args.min_pass for s in summary.values()):
        print(f"FAILED: pass rate below {args.min_pass:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
