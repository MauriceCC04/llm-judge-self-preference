"""tools/render_frozen_docs.py — print the frozen study summary block for docs."""
from __future__ import annotations

from generate.study_manifest import render_study_summary_markdown


def main() -> None:
    print(render_study_summary_markdown())


if __name__ == "__main__":
    main()
