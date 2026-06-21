from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8001")
EVAL_SET_PATH = os.environ.get("EVAL_SET_PATH", str(ROOT / "evals" / "eval_set.jsonl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run questions against the LangGraph agent /answer endpoint")
    parser.add_argument("--samples", type=int, default=10, help="Number of eval_set questions to send (default: 10)")
    parser.add_argument("--question", help="Ask a single ad-hoc question (requires --db; ignores --samples)")
    parser.add_argument("--db", help="Database id for --question, e.g. financial")
    parser.add_argument("--agent-url", default=AGENT_URL, help="Base URL of the agent server")
    return parser.parse_args()


def ask(client: httpx.Client, base_url: str, question: str, db: str) -> dict:
    resp = client.post(f"{base_url}/answer", json={"question": question, "db": db})
    resp.raise_for_status()
    return resp.json()


def print_result(idx: int, total: int, question: str, db: str, data: dict) -> None:
    print(f"[{idx}/{total}] {db}: {question}")
    print(f"    ok={data.get('ok')}  iterations={data.get('iterations')}")
    print(f"    sql: {data.get('sql', '')}")
    if data.get("ok"):
        rows = data.get("rows") or []
        print(f"    rows: {len(rows)} returned" + (f" -> {rows[:3]}" if rows else ""))
    else:
        print(f"    error: {data.get('error')}")
    print()


def main() -> None:
    args = parse_args()
    base_url = args.agent_url.rstrip("/")

    if args.question:
        if not args.db:
            print("❌ --question requires --db", file=sys.stderr)
            sys.exit(1)
        jobs = [{"question": args.question, "db_id": args.db}]
    else:
        eval_file = Path(EVAL_SET_PATH)
        if not eval_file.exists():
            print(f"❌ {eval_file} not found - run scripts/load_data.py first", file=sys.stderr)
            sys.exit(1)
        questions = [json.loads(line) for line in eval_file.read_text().splitlines() if line.strip()]
        jobs = questions[: args.samples]

    print(f"🚀 Querying agent at {base_url} ({len(jobs)} question(s))\n")

    with httpx.Client(timeout=120.0) as client:
        try:
            client.get(f"{base_url}/health").raise_for_status()
        except httpx.HTTPError as e:
            print(f"❌ Agent not reachable at {base_url}: {e}", file=sys.stderr)
            sys.exit(1)

        ok_count = 0
        for i, q in enumerate(jobs, 1):
            try:
                data = ask(client, base_url, q["question"], q["db_id"])
                ok_count += int(bool(data.get("ok")))
                print_result(i, len(jobs), q["question"], q["db_id"], data)
            except httpx.HTTPStatusError as e:
                print(f"[{i}/{len(jobs)}] {q['db_id']}: HTTP {e.response.status_code} - {e.response.text[:160]}\n")
            except httpx.HTTPError as e:
                print(f"[{i}/{len(jobs)}] {q['db_id']}: ERROR {type(e).__name__}: {e}\n")

    print(f"Done. {ok_count}/{len(jobs)} returned ok=true.")


if __name__ == "__main__":
    main()

