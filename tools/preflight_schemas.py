from __future__ import annotations

import argparse
import importlib
import json
from typing import Any, Callable

from trailtraining.llm.schemas import (
    MACHINE_PLAN_SCHEMA,
    PLAN_EXPLANATION_SCHEMA,
    TRAINING_PLAN_SCHEMA,
)

_SCHEMAS: tuple[dict[str, Any], ...] = (
    MACHINE_PLAN_SCHEMA,
    PLAN_EXPLANATION_SCHEMA,
    TRAINING_PLAN_SCHEMA,
)


def _invoke_compiler(
    func: Callable[..., Any],
    schema_obj: dict[str, Any],
    schema_text: str,
) -> None:
    last_type_error: TypeError | None = None
    attempts: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
        ((schema_obj,), {}),
        ((schema_text,), {}),
        ((), {"schema": schema_obj}),
        ((), {"schema": schema_text}),
        ((), {"json_schema": schema_obj}),
        ((), {"json_schema": schema_text}),
    )

    for args, kwargs in attempts:
        try:
            func(*args, **kwargs)
            return
        except TypeError as exc:
            last_type_error = exc
            continue
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    if last_type_error is not None:
        raise RuntimeError(
            f"No compatible xgrammar compiler signature found for {func!r}: {last_type_error}"
        ) from last_type_error
    raise RuntimeError(f"Could not invoke xgrammar compiler function {func!r}.")


def _compile_with_xgrammar(schema_obj: dict[str, Any]) -> str:
    schema_text = json.dumps(schema_obj, ensure_ascii=False)

    try:
        xgrammar = importlib.import_module("xgrammar")
    except Exception as exc:
        raise RuntimeError(
            "xgrammar is not importable in this environment. "
            "If vLLM is installed correctly, xgrammar should usually be available as a transitive dependency."
        ) from exc

    compiler_cls = getattr(xgrammar, "GrammarCompiler", None)
    if callable(compiler_cls):
        compiler = compiler_cls()
        for method_name in (
            "compile_json_schema",
            "compile_schema",
            "compile_json_schema_string",
        ):
            method = getattr(compiler, method_name, None)
            if callable(method):
                _invoke_compiler(method, schema_obj, schema_text)
                return f"xgrammar.GrammarCompiler.{method_name}"

    grammar_cls = getattr(xgrammar, "Grammar", None)
    if grammar_cls is not None:
        for method_name in (
            "from_json_schema",
            "from_schema",
            "compile_json_schema",
        ):
            method = getattr(grammar_cls, method_name, None)
            if callable(method):
                _invoke_compiler(method, schema_obj, schema_text)
                return f"xgrammar.Grammar.{method_name}"

    raise RuntimeError(
        "Found xgrammar, but could not locate a supported JSON-schema compilation entrypoint."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-fast schema preflight before long generation runs."
    )
    parser.add_argument(
        "--backend",
        default="xgrammar",
        choices=("xgrammar", "outlines"),
        help="Guided decoding backend that the job will use.",
    )
    args = parser.parse_args()

    for schema in _SCHEMAS:
        name = str(schema.get("name") or "unnamed_schema")
        body = schema.get("schema")
        if not isinstance(body, dict):
            raise RuntimeError(f"Schema '{name}' is missing a dict-valued 'schema' body.")

        json.dumps(body, ensure_ascii=False)

        if args.backend == "xgrammar":
            method_used = _compile_with_xgrammar(body)
            print(f"[OK] {name}: compiled with {method_used}")
        else:
            print(
                f"[SKIP] {name}: JSON-serializable; xgrammar preflight skipped because backend=outlines"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
