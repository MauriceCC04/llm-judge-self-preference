from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


class BadFixturePromptHygieneTests(unittest.TestCase):
    def _fixture_dir(self) -> Path:
        raw = os.getenv("JBS_BAD_FIXTURE_DIR", "").strip()
        if not raw:
            raise unittest.SkipTest("Set JBS_BAD_FIXTURE_DIR to the bad fixture directory to run this test.")
        path = Path(raw)
        if not path.exists():
            raise unittest.SkipTest(f"Fixture dir does not exist: {path}")
        return path

    def _saved_plan(self) -> Path | None:
        raw = os.getenv("JBS_BAD_SAVED_PLAN", "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.exists():
            raise unittest.SkipTest(f"Saved plan path does not exist: {path}")
        return path

    def _machine_plan(self) -> Path | None:
        raw = os.getenv("JBS_BAD_MACHINE_PLAN", "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.exists():
            raise unittest.SkipTest(f"Machine plan path does not exist: {path}")
        return path

    def test_bad_fixture_prompt_hygiene(self) -> None:
        try:
            import replay_bad_fixture_locally as replay
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"Could not import replay_bad_fixture_locally: {exc}") from exc

        fixture_dir = self._fixture_dir()
        saved_plan = self._saved_plan()
        machine_plan = self._machine_plan()
        if saved_plan is None and machine_plan is None:
            raise unittest.SkipTest("Set JBS_BAD_SAVED_PLAN or JBS_BAD_MACHINE_PLAN for prompt replay.")

        with tempfile.TemporaryDirectory(prefix="jbs_bad_fixture_") as tmpdir:
            out_dir = Path(tmpdir)
            rc = replay.build_replay_artifacts(
                fixture_dir=fixture_dir,
                out_dir=out_dir,
                saved_plan_json=saved_plan,
                machine_plan_json=machine_plan,
                max_chars=20000,
                detail_days=None,
            )

            audit = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 0, msg=json.dumps(audit, indent=2, ensure_ascii=False))
            self.assertEqual(audit["counts"]["prompt_hits"], 0, msg=json.dumps(audit, indent=2, ensure_ascii=False))
            self.assertEqual(audit["counts"]["signal_registry_hits"], 0, msg=json.dumps(audit, indent=2, ensure_ascii=False))
            self.assertEqual(audit["counts"]["machine_plan_hits"], 0, msg=json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
