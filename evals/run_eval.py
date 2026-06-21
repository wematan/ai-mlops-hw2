"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question end-to-end."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    q_text = question["question"]

    try:
        resp = httpx.post(
            agent_url, json={"question": q_text, "db": db_id}, timeout=120.0
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            "db_id": db_id,
            "question": q_text,
            "gold_sql": gold_sql,
            "agent_sql": "",
            "iterations": 0,
            "agent_ok": False,
            "correct": False,
            "per_iteration_correct": [],
            "error": f"agent call failed: {type(e).__name__}: {e}",
        }

    agent_sql = data.get("sql") or ""
    iterations = int(data.get("iterations") or 0)
    agent_ok = bool(data.get("ok"))
    history = data.get("history") or []

    # The SQL the agent produced at each attempt, in order. Verify steps carry
    # no SQL, so only generate_sql / revise entries count as attempts.
    attempts = [
        h["sql"]
        for h in history
        if h.get("node") in ("generate_sql", "revise") and h.get("sql")
    ]
    if not attempts and agent_sql:
        attempts = [agent_sql]

    # Gold rows once; reused to score every attempt.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    per_iteration_correct: list[bool] = []
    for sql in attempts:
        ok, rows, _ = run_sql(db_id, sql)
        per_iteration_correct.append(bool(gold_ok and ok and matches(gold_rows, rows)))

    return {
        "db_id": db_id,
        "question": q_text,
        "gold_sql": gold_sql,
        "agent_sql": agent_sql,
        "iterations": iterations,
        "agent_ok": agent_ok,
        "correct": per_iteration_correct[-1] if per_iteration_correct else False,
        "per_iteration_correct": per_iteration_correct,
        "gold_error": None if gold_ok else gold_err,
        "error": None,
    }


def _correct_at(per_iteration: list[bool], k: int) -> bool:
    """Correctness at iteration k, carried forward past early termination."""
    if not per_iteration:
        return False
    return per_iteration[k] if k < len(per_iteration) else per_iteration[-1]


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n": 0, "overall_pass_rate": 0.0, "pass_rate_by_iteration": []}

    def rate(count: int) -> float:
        return round(count / n, 4)

    # Per-iteration pass rate with carry-forward, indexed by attempt (iter 0 =
    # first generate_sql, iter 1 = after first revise, ...).
    max_iters = max(
        (len(r.get("per_iteration_correct") or []) for r in results), default=0
    )
    pass_rate_by_iteration = [
        rate(sum(1 for r in results if _correct_at(r.get("per_iteration_correct") or [], k)))
        for k in range(max_iters)
    ]

    # Did the revise loop earn its keep, per question?
    fixed_by_revision = 0
    broken_by_revision = 0
    for r in results:
        pic = r.get("per_iteration_correct") or []
        if not pic:
            continue
        first, final = pic[0], bool(r.get("correct"))
        if not first and final:
            fixed_by_revision += 1
        elif first and not final:
            broken_by_revision += 1

    iters = [int(r.get("iterations") or 0) for r in results]
    histogram: dict[str, int] = {}
    for it in iters:
        histogram[str(it)] = histogram.get(str(it), 0) + 1

    return {
        "n": n,
        "overall_pass_rate": rate(sum(1 for r in results if r.get("correct"))),
        "pass_rate_by_iteration": pass_rate_by_iteration,
        "agent_ok_rate": rate(sum(1 for r in results if r.get("agent_ok"))),
        "fixed_by_revision": fixed_by_revision,
        "broken_by_revision": broken_by_revision,
        "mean_iterations": round(sum(iters) / n, 3),
        "iterations_histogram": dict(sorted(histogram.items())),
        "errors": sum(1 for r in results if r.get("error")),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
