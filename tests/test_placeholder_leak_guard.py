from __future__ import annotations

import unittest
from pathlib import Path


def _import_guard_module():
    try:
        from generate import trailtraining_compat as mod  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise unittest.SkipTest(f"Could not import generate.trailtraining_compat: {exc}") from exc
    return mod


class PlaceholderLeakGuardTests(unittest.TestCase):
    def test_collect_detects_nested_snapshot_leaks(self) -> None:
        mod = _import_guard_module()
        collect = getattr(mod, "_collect_placeholder_leaks", None)
        if collect is None:
            self.fail("Missing _collect_placeholder_leaks in generate.trailtraining_compat")

        obj = {
            "snapshot": {
                "last7": {
                    "distance_km": ">{signal_id: ",
                    "moving_time_hours": "4.0",
                },
                "baseline28": {
                    "distance_km": "30",
                    "moving_time_hours": ">{signal_id: ",
                },
            }
        }
        hits = collect(obj)
        self.assertGreaterEqual(len(hits), 2)
        paths = {path for path, _ in hits}
        self.assertIn("$.snapshot.last7.distance_km", paths)
        self.assertIn("$.snapshot.baseline28.moving_time_hours", paths)

    def test_assert_rejects_corrupted_artifact(self) -> None:
        mod = _import_guard_module()
        guard = getattr(mod, "_assert_no_placeholder_leaks", None)
        if guard is None:
            self.fail("Missing _assert_no_placeholder_leaks in generate.trailtraining_compat")

        obj = {
            "meta": {"today": "2026-03-17"},
            "snapshot": {
                "last7": {
                    "distance_km": ">{signal_id: ",
                    "moving_time_hours": "4.0",
                    "elevation_m": "500",
                    "activity_count": "5",
                    "sleep_hours_mean": "7.5",
                    "hrv_mean": "68",
                    "rhr_mean": "46",
                },
                "baseline28": {
                    "distance_km": "30",
                    "moving_time_hours": "4.0",
                    "elevation_m": "500",
                    "activity_count": "5",
                    "sleep_hours_mean": "7.5",
                    "hrv_mean": "68",
                    "rhr_mean": "46",
                },
                "notes": "Bad artifact",
            },
        }

        with self.assertRaises(Exception):
            guard(plan_id="bad_fixture", output_path=Path("/tmp/bad_fixture.json"), plan_obj=obj)

    def test_assert_allows_clean_artifact(self) -> None:
        mod = _import_guard_module()
        guard = getattr(mod, "_assert_no_placeholder_leaks", None)
        if guard is None:
            self.fail("Missing _assert_no_placeholder_leaks in generate.trailtraining_compat")

        obj = {
            "meta": {"today": "2026-03-17"},
            "snapshot": {
                "last7": {
                    "distance_km": "26.7",
                    "moving_time_hours": "3.7",
                    "elevation_m": "835.5",
                    "activity_count": "4",
                    "sleep_hours_mean": "6.36",
                    "hrv_mean": "58",
                    "rhr_mean": "49",
                },
                "baseline28": {
                    "distance_km": "110",
                    "moving_time_hours": "17.5",
                    "elevation_m": "2400",
                    "activity_count": "17",
                    "sleep_hours_mean": "6.8",
                    "hrv_mean": "61",
                    "rhr_mean": "48",
                },
                "notes": "Clean artifact",
            },
        }

        guard(plan_id="clean_fixture", output_path=Path("/tmp/clean_fixture.json"), plan_obj=obj)


if __name__ == "__main__":
    unittest.main()
