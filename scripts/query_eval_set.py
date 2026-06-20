from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agent.schema import render_schema
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
EVAL_SET_PATH = os.environ.get("EVAL_SET_PATH", str(ROOT / "evals" / "eval_set.jsonl"))
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "0"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0" if THINKING_BUDGET > 0 else "0"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BIRD eval prompts against a vLLM OpenAI-compatible server")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to run from eval_set (default: all)",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    eval_set_file = Path(EVAL_SET_PATH)
    if not eval_set_file.exists():
        print(f"❌ {eval_set_file} not found", file=sys.stderr)
        sys.exit(1)

    questions = [json.loads(line) for line in eval_set_file.read_text().splitlines() if line.strip()]
    if not questions:
        print(f"❌ {eval_set_file} is empty", file=sys.stderr)
        sys.exit(1)

    if args.samples is not None:
        if args.samples < 0:
            print("❌ --samples must be >= 0", file=sys.stderr)
            sys.exit(1)
        questions = questions[: args.samples]

    print(f"📊 Loaded {len(questions)} questions from {eval_set_file}")
    print(f"🚀 Querying vLLM at {VLLM_BASE_URL}")
    print(f"   Model: {VLLM_MODEL}")
    print(f"   Max tokens: {MAX_TOKENS}")
    print(f"   Thinking enabled: {THINKING_BUDGET > 0}")
    print(f"   Temperature: {TEMPERATURE}")
    print(f"   Stop tokens: ['<think>', '</think>']\n")

    client = OpenAI(api_key=LLM_API_KEY, base_url=VLLM_BASE_URL)


    for i, q in enumerate(questions, 1):
        question = q.get("question")
        db_id = q.get("db_id")
        gold_sql = q.get("gold_sql")

        if not question or not db_id or not gold_sql:
            print(f"❌ [{i}/{len(questions)}] malformed record: missing question/db_id/gold_sql")
            print()
            continue

        try:
            # Build request parameters
            schema = render_schema(db_id)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a SQLite expert. Schema:\n"
                        f"{schema}\n"
                        "Output only one SQL query, no prose."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]
            
            # Prepare extra body for thinking if enabled
            thinking_enabled = THINKING_BUDGET > 0
            if thinking_enabled:
                extra_body = {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": THINKING_BUDGET,
                    },
                }
            else:
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            
            response = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stop=["<think>", "</think>"],
                extra_body=extra_body,
            )
            
            # Extract thinking (if present) and response
            thinking_content = ""
            response_content = ""
            
            content = response.choices[0].message.content
            if isinstance(content, str):
                response_content = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            thinking_content = block.get("thinking", "")
                        elif block.get("type") == "text":
                            response_content = block.get("text", "")

            print(f"[{i}/{len(questions)}] {db_id} [thinking={thinking_enabled}]")
            print(f"   Q: {question}")
            if thinking_content:
                print(f"   💭 Thinking: {thinking_content}")
            print(f"   A: {response_content}")
            print(f"   GT: {gold_sql}")
            print()

        except Exception as e:  # noqa: BLE001
            print(f"❌ [{i}/{len(questions)}] {db_id}: {type(e).__name__}: {e}")
            print()



if __name__ == "__main__":
    main()

