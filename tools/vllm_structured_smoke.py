from __future__ import annotations

import argparse
import json
import os
from typing import Any

from openai import OpenAI

from trailtraining.llm.schemas import MACHINE_PLAN_SCHEMA
from trailtraining.llm.shared import call_with_schema

_SIMPLE_SCHEMA: dict[str, Any] = {
    "name": "trailtraining_vllm_smoke_simple_v1",
    "schema": {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "label": {"type": "string"},
        },
        "required": ["ok", "label"],
        "additionalProperties": False,
    },
}


def _simple_prompt() -> str:
    return (
        "Return a JSON object with exactly these fields: "
        "ok=true and label='smoke-pass'. Return only JSON."
    )


def _machine_prompt() -> str:
    return (
        "Return ONLY valid JSON matching the provided schema. "
        "Create a 1-day trailrunning machine plan with one easy session, no extra commentary."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test local vLLM structured outputs.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--schema",
        choices=("simple", "machine_plan"),
        default="simple",
        help="Schema to test against the local vLLM server.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable one-shot structured-output request/response debug logging.",
    )
    args = parser.parse_args()

    if args.debug:
        os.environ["TRAILTRAINING_DEBUG_STRUCTURED_OUTPUTS"] = "1"

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key="dummy")
    if args.schema == "simple":
        schema = _SIMPLE_SCHEMA
        prompt = _simple_prompt()
    else:
        schema = MACHINE_PLAN_SCHEMA
        prompt = _machine_prompt()

    kwargs: dict[str, Any] = {
        "model": args.model,
        "instructions": "You are a careful assistant that returns only JSON.",
        "input": prompt,
        "temperature": 0.0,
    }
    response = call_with_schema(client, kwargs, schema)
    raw_text = getattr(response, "output_text", None) or str(response)
    print(f"schema={schema['name']}")
    print(f"raw_text={raw_text}")

    parsed = json.loads(raw_text)
    print("parsed_ok=true")

    if args.schema == "simple":
        assert isinstance(parsed, dict), parsed
        assert parsed.get("ok") is True, parsed
        assert isinstance(parsed.get("label"), str) and parsed.get("label"), parsed
        print("simple_schema_ok=true")
    else:
        assert isinstance(parsed, dict)
        print("project_schema_json_ok=true")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())