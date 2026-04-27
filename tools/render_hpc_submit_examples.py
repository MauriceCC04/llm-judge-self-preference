"""tools/render_hpc_submit_examples.py — render safe wrapped sbatch commands.

This helper bakes in the operational lessons from Bocconi's `stud` partition:

- prefer wrapped `sbatch` commands with an absolute repo path
- avoid ambiguous `$PROJECT_ROOT` examples in user-facing docs
- expose per-judge buffered walltimes
- provide sequential judge-batch planning instead of encouraging a burst of
  loosely-related submissions on `stud`
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

DEFAULT_PARTITION = "stud"
DEFAULT_QOS = "stud"
DEFAULT_EXCLUDE_NODE = "gnode04"

try:
    from generate.constants import ACTIVE_JUDGE_NAMES
    from judge.panel import get_judge, walltime_hours_with_buffer
except Exception:  # pragma: no cover - keep helper usable in isolation.
    ACTIVE_JUDGE_NAMES = [
        "llama_8b_judge",
        "qwen_7b_judge",
        "qwen_14b_judge",
        "qwen_32b_judge",
    ]

    class _FallbackJudge:
        def __init__(self, name: str) -> None:
            self.name = name
            self.time_hours = 6.0

    def get_judge(name: str) -> _FallbackJudge:
        return _FallbackJudge(name)

    def walltime_hours_with_buffer(judge: _FallbackJudge, *, multiplier: float = 1.2) -> int:
        return int(judge.time_hours * multiplier)


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _wrap(repo_root: Path, script: str, exports: dict[str, str] | None = None) -> str:
    commands: list[str] = []
    for key, value in (exports or {}).items():
        commands.append(f"export {key}={_quote(value)}")
    commands.append(f"cd {_quote(repo_root)}")
    commands.append(f"bash {_quote(script)}")
    return " && ".join(commands)


def _sbatch_command(
    *,
    wrap: str,
    job_name: str,
    account: str,
    email: str,
    partition: str = DEFAULT_PARTITION,
    qos: str = DEFAULT_QOS,
    exclude_node: str = DEFAULT_EXCLUDE_NODE,
    extra_flags: list[str] | None = None,
) -> str:
    parts = [
        "sbatch",
        f"--job-name={_quote(job_name)}",
        f"--account={_quote(account)}",
        f"--partition={_quote(partition)}",
        f"--qos={_quote(qos)}",
        f"--exclude={_quote(exclude_node)}",
        "--mail-type=END,FAIL",
        f"--mail-user={_quote(email)}",
        "--output=out/%x_%j.out",
        "--error=err/%x_%j.err",
    ]
    parts.extend(extra_flags or [])
    parts.append(f"--wrap={_quote(wrap)}")
    return " \\\n  ".join(parts)


def _judge_buffered_hours(judge_name: str) -> int:
    judge = get_judge(judge_name)
    return walltime_hours_with_buffer(judge)


def _chunk_judges_by_hours(judges: list[str], max_batch_hours: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_hours = 0

    for judge_name in judges:
        judge_hours = _judge_buffered_hours(judge_name)
        if current and current_hours + judge_hours > max_batch_hours:
            batches.append(current)
            current = []
            current_hours = 0
        current.append(judge_name)
        current_hours += judge_hours

    if current:
        batches.append(current)
    return batches


def _render_sequential_launcher_script(judges: list[str]) -> str:
    quoted = " ".join(_quote(name) for name in judges)
    return f"""#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${{BASH_SOURCE[0]}}")/common.sh"
activate_env

cd "${{PROJECT_ROOT}}"
mkdir -p out err judgments results

JUDGES=({quoted})
PAIRWISE_VIEW="${{PAIRWISE_VIEW:-raw_normalized}}"
JUDGE_MODE="${{JUDGE_MODE:-full}}"
RUN_PAIRWISE="${{RUN_PAIRWISE:-1}}"
RUN_SOFT_EVAL="${{RUN_SOFT_EVAL:-1}}"
REQUIRE_STYLE_GATE="${{REQUIRE_STYLE_GATE:-1}}"
STYLE_GATE_SUMMARY="${{STYLE_GATE_SUMMARY:-results/style_audit_summary.json}}"

for JUDGE_NAME in "${{JUDGES[@]}}"; do
  echo "=== Sequential judge batch: ${{JUDGE_NAME}} ==="
  bash slurm/pre_cache_models.sh "${{JUDGE_NAME}}"
  export JUDGE_NAME PAIRWISE_VIEW JUDGE_MODE RUN_PAIRWISE RUN_SOFT_EVAL REQUIRE_STYLE_GATE STYLE_GATE_SUMMARY
  bash slurm/run_judge_hpc.sh
done
"""


def render_commands(
    repo_root: str,
    *,
    account: str,
    email: str,
    exclude_node: str = DEFAULT_EXCLUDE_NODE,
    max_sequential_batch_hours: int = 24,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()

    commands: dict[str, Any] = {
        "repo_root": str(root),
        "common_sbatch_flags": {
            "account": account,
            "partition": DEFAULT_PARTITION,
            "qos": DEFAULT_QOS,
            "exclude": exclude_node,
            "mail_type": "END,FAIL",
            "mail_user": email,
        },
        "notes": [
            "Use absolute repo-root wraps on Bocconi rather than relying on $PROJECT_ROOT in the submitting shell.",
            "Prefer a small number of sequential judge batches on `stud` instead of many loosely-related submissions.",
            "The sequential launcher template assumes slurm/run_judge_hpc.sh is the real full judge runner.",
        ],
    }

    commands["preflight"] = _sbatch_command(
        wrap=_wrap(root, "slurm/run_preflight.sh"),
        job_name="judge-bias-preflight",
        account=account,
        email=email,
        exclude_node=exclude_node,
    )

    commands["smoke"] = _sbatch_command(
        wrap=_wrap(root, "slurm/run_vllm_smoke.sh"),
        job_name="judge-bias-vllm-smoke",
        account=account,
        email=email,
        exclude_node=exclude_node,
    )

    commands["generation"] = {
        "llm_llama": _sbatch_command(
            wrap=_wrap(
                root,
                "slurm/run_generation_hpc.sh",
                exports={
                    "GENERATION_ARM": "llm",
                    "GENERATION_PROFILE": "exact",
                    "LLM_SOURCE_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
                },
            ),
            job_name="judge-bias-gen-llama",
            account=account,
            email=email,
            exclude_node=exclude_node,
        ),
        "llm_qwen": _sbatch_command(
            wrap=_wrap(
                root,
                "slurm/run_generation_hpc.sh",
                exports={
                    "GENERATION_ARM": "llm",
                    "GENERATION_PROFILE": "exact",
                    "LLM_SOURCE_MODEL": "Qwen/Qwen2.5-7B-Instruct",
                },
            ),
            job_name="judge-bias-gen-qwen",
            account=account,
            email=email,
            exclude_node=exclude_node,
        ),
        "programmatic": _sbatch_command(
            wrap=_wrap(
                root,
                "slurm/run_generation_hpc.sh",
                exports={
                    "GENERATION_ARM": "programmatic",
                    "GENERATION_PROFILE": "exact",
                    "SAMPLER_CONFIG": "sampler_config.json",
                },
            ),
            job_name="judge-bias-gen-prog",
            account=account,
            email=email,
            exclude_node=exclude_node,
        ),
    }

    per_judge: dict[str, Any] = {}
    for judge_name in ACTIVE_JUDGE_NAMES:
        walltime = f"{_judge_buffered_hours(judge_name):02d}:00:00"
        per_judge[judge_name] = {
            "walltime": walltime,
            "full": _sbatch_command(
                wrap=_wrap(
                    root,
                    "slurm/run_judge_hpc.sh",
                    exports={
                        "JUDGE_NAME": judge_name,
                        "JUDGE_MODE": "full",
                        "PAIRWISE_VIEW": "raw_normalized",
                    },
                ),
                job_name=f"judge-{judge_name}",
                account=account,
                email=email,
                exclude_node=exclude_node,
                extra_flags=[f"--time={walltime}"],
            ),
            "pilot": _sbatch_command(
                wrap=_wrap(
                    root,
                    "slurm/run_judge_hpc.sh",
                    exports={
                        "JUDGE_NAME": judge_name,
                        "JUDGE_MODE": "pilot",
                        "PAIRWISE_VIEW": "raw_normalized",
                    },
                ),
                job_name=f"judge-pilot-{judge_name}",
                account=account,
                email=email,
                exclude_node=exclude_node,
                extra_flags=[f"--time={walltime}"],
            ),
            "masked_view": _sbatch_command(
                wrap=_wrap(
                    root,
                    "slurm/run_judge_hpc.sh",
                    exports={
                        "JUDGE_NAME": judge_name,
                        "JUDGE_MODE": "full",
                        "PAIRWISE_VIEW": "canonical_masked",
                    },
                ),
                job_name=f"judge-masked-{judge_name}",
                account=account,
                email=email,
                exclude_node=exclude_node,
                extra_flags=[f"--time={walltime}"],
            ),
        }
    commands["per_judge"] = per_judge

    sequential_batches: list[dict[str, Any]] = []
    for idx, batch in enumerate(
        _chunk_judges_by_hours(list(ACTIVE_JUDGE_NAMES), max_sequential_batch_hours),
        start=1,
    ):
        batch_hours = sum(_judge_buffered_hours(name) for name in batch)
        launcher_name = f"slurm/run_judge_panel_sequential_batch_{idx:02d}.sh"
        sequential_batches.append(
            {
                "batch_name": f"judge-panel-batch-{idx:02d}",
                "judges": batch,
                "buffered_walltime_hours": batch_hours,
                "launcher_script_path": launcher_name,
                "launcher_script_body": _render_sequential_launcher_script(batch),
                "sbatch_command": _sbatch_command(
                    wrap=_wrap(root, launcher_name),
                    job_name=f"judge-panel-batch-{idx:02d}",
                    account=account,
                    email=email,
                    exclude_node=exclude_node,
                    extra_flags=[f"--time={batch_hours:02d}:00:00"],
                ),
            }
        )
    commands["sequential_batches"] = sequential_batches
    return commands


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render safe wrapped sbatch examples for Bocconi HPC."
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--exclude-node", default=DEFAULT_EXCLUDE_NODE)
    parser.add_argument("--max-sequential-batch-hours", type=int, default=24)
    args = parser.parse_args(argv)

    report = render_commands(
        args.repo_root,
        account=args.account,
        email=args.email,
        exclude_node=args.exclude_node,
        max_sequential_batch_hours=args.max_sequential_batch_hours,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()