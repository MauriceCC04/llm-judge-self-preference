"""tools/render_hpc_submit_examples.py — render safe wrapped sbatch commands.

The goal is to remove ambiguous $PROJECT_ROOT examples from the runbook and to
promote one sequential judge launcher plan instead of many loosely-related
submissions on the `stud` QoS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from generate.constants import ACTIVE_JUDGE_NAMES
except Exception:  # pragma: no cover - keep this helper usable in isolation.
    ACTIVE_JUDGE_NAMES = ["llama_8b", "qwen_7b", "qwen_14b", "qwen_32b"]



def _wrap(repo_root: str, script: str, exports: dict[str, str] | None = None) -> str:
    export_bits = []
    for key, value in (exports or {}).items():
        export_bits.append(f"export {key}={value}")
    prefix = " && ".join(export_bits)
    body = f"cd {repo_root} && bash {script}"
    return f"{prefix} && {body}" if prefix else body



def render_commands(repo_root: str, *, account: str, email: str, exclude_node: str = "gnode04") -> dict[str, Any]:
    root = str(Path(repo_root).expanduser())
    preflight = {
        "job_name": "judge-bias-preflight",
        "command": _wrap(root, "slurm/run_preflight.sh"),
    }
    smoke = {
        "job_name": "judge-bias-vllm-smoke",
        "command": _wrap(root, "slurm/run_vllm_smoke.sh"),
    }
    sequential_panel = {
        "job_name": "judge-bias-full-panel-sequential",
        "judges": list(ACTIVE_JUDGE_NAMES),
        "strategy": "Run one judge after another; do not mass-submit on stud.",
        "template_exports": {
            "JUDGES": '"' + " ".join(ACTIVE_JUDGE_NAMES) + '"',
            "PAIRWISE_VIEW": "canonical_masked",
            "JUDGE_MODE": "full",
        },
        "recommended_wrapper": (
            "Create a single sequential launcher that loops over the judges and calls "
            "slurm/run_judge_hpc.sh for each exported judge name."
        ),
    }
    common_flags = {
        "account": account,
        "partition": "stud",
        "qos": "stud",
        "exclude": exclude_node,
        "mail_type": "END,FAIL",
        "mail_user": email,
    }
    return {
        "repo_root": root,
        "common_sbatch_flags": common_flags,
        "preflight": preflight,
        "smoke": smoke,
        "sequential_panel": sequential_panel,
    }



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render safe wrapped sbatch examples for Bocconi HPC.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--exclude-node", default="gnode04")
    args = parser.parse_args(argv)
    report = render_commands(
        args.repo_root,
        account=args.account,
        email=args.email,
        exclude_node=args.exclude_node,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
