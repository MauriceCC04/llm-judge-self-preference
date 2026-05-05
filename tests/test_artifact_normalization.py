from __future__ import annotations

import copy

from generate.artifact_normalization import (
    CANONICAL_TITLES,
    audit_plan_presentation,
    artifact_sha256,
    attach_normalization_metadata_to_provenance,
    build_normalization_provenance_payload,
    normalize_plan_artifact,
    score_relevant_projection,
)


def _day(session_type: str, title: str = "Day 1", workout: str | None = None, purpose: str | None = None, **extra):
    is_rest = session_type == "rest"
    return {
        "date": extra.pop("date", "2026-03-18"),
        "title": title,
        "session_type": session_type,
        "is_rest_day": is_rest,
        "is_hard_day": session_type in {"tempo", "intervals", "hills"},
        "duration_minutes": 0 if is_rest else 45,
        "target_intensity": "rest" if is_rest else "moderate",
        "workout": workout if workout is not None else f"{title}. Placeholder.",
        "purpose": purpose if purpose is not None else "Support the week.",
        **extra,
    }


def _plan(days):
    return {
        "meta": {"plan_days": 7, "style": "trailrunning"},
        "plan": {"weekly_totals": {"planned_moving_time_hours": 5.0}, "days": days},
    }


def test_canonicalizes_titles_for_known_session_types():
    days = []
    for idx, session_type in enumerate(CANONICAL_TITLES):
        days.append(_day(session_type, title=f"Day {idx + 1}", date=f"2026-03-{18 + idx:02d}"))
    plan = _plan(days)

    result = normalize_plan_artifact(plan, expected_days=len(days))

    assert not result.errors
    assert [d["title"] for d in result.plan["plan"]["days"]] == list(CANONICAL_TITLES.values())
    assert all("generic_day_title" not in issue for issue in audit_plan_presentation(result.plan, expected_days=len(days)))


def test_repairs_clear_workout_and_purpose_contradictions_without_structural_changes():
    plan = _plan(
        [
            _day("long", title="Cross-training", workout="Cross-training. Use the bike.", purpose="Strength session. Build legs."),
            _day("strength", title="Long run", workout="Long run. Build endurance.", purpose="Tempo run. Improve speed."),
            _day("cross", title="Long run", workout="Long run. Build endurance.", purpose="Long run. Build endurance."),
            _day("tempo", title="Day 4", workout="Tempo run. Improve rhythm.", purpose="Strength session. Build legs."),
            _day("easy", title="Day 5", workout="Easy run. Keep it comfortable.", purpose="Aerobic support."),
            _day("rest", title="Day 6"),
            _day("rest", title="Rest day"),
        ]
    )
    before = score_relevant_projection(plan)

    result = normalize_plan_artifact(plan)

    assert not result.errors
    assert result.structural_projection_unchanged
    assert score_relevant_projection(result.plan) == before
    assert result.plan["plan"]["days"][0]["title"] == "Long run"
    assert result.plan["plan"]["days"][1]["title"] == "Strength session"
    assert result.plan["plan"]["days"][2]["title"] == "Cross-training"
    assert "Strength session" not in result.plan["plan"]["days"][3]["purpose"]
    assert not audit_plan_presentation(result.plan)


def test_repairs_rest_described_as_active_and_active_described_as_rest():
    plan = _plan(
        [
            _day("rest", title="Rest day", workout="Easy run for 30 minutes.", purpose="Run to loosen up."),
            _day("easy", title="Rest day", workout="Rest day. No structured training.", purpose="Rest day after work."),
            _day("aerobic", title="Day 3", workout="Aerobic run. Keep it steady."),
            _day("long", title="Day 4", workout="Long run. Build endurance."),
            _day("tempo", title="Day 5", workout="Tempo run. Controlled hard effort."),
            _day("strength", title="Day 6", workout="Strength session. Mobility and lifting."),
            _day("cross", title="Day 7", workout="Cross-training. Easy bike."),
        ]
    )

    result = normalize_plan_artifact(plan)

    assert not result.errors
    assert result.plan["plan"]["days"][0]["workout"] == "Rest day. No structured training."
    assert result.plan["plan"]["days"][1]["workout"] == "Easy run at comfortable conversational effort."
    assert not audit_plan_presentation(result.plan)


def test_idempotent_and_provenance_metadata_is_sidecar_only():
    plan = _plan(
        [
            _day("easy", title="Day 1"),
            _day("aerobic", title="Trail Run"),
            _day("long", title="Aerobic run", workout="Long run. Build endurance."),
            _day("tempo", title="Day 4", workout="Tempo run. Improve rhythm."),
            _day("intervals", title="Day 5", workout="Intervals. Develop pacing."),
            _day("strength", title="Day 6", workout="Strength session. Mobility."),
            _day("rest", title="Day 7"),
        ]
    )
    first = normalize_plan_artifact(plan)
    second = normalize_plan_artifact(first.plan)

    assert first.changed
    assert not second.changed
    assert second.plan == first.plan

    payload = build_normalization_provenance_payload(
        original_sha256=artifact_sha256(plan),
        normalized_sha256=artifact_sha256(first.plan),
        result=first,
        timestamp="2026-05-05T00:00:00+00:00",
    )
    provenance = attach_normalization_metadata_to_provenance({"runtime_metadata": {}}, payload)
    assert "artifact_normalization" in provenance["runtime_metadata"]
    assert "artifact_normalization" not in first.plan


def test_active_zero_duration_is_warned_not_structurally_edited():
    plan = _plan(
        [
            _day("easy", title="Day 1", duration_minutes=0),
            _day("easy", title="Day 2"),
            _day("easy", title="Day 3"),
            _day("easy", title="Day 4"),
            _day("easy", title="Day 5"),
            _day("easy", title="Day 6"),
            _day("easy", title="Day 7"),
        ]
    )
    before = copy.deepcopy(plan["plan"]["days"][0])

    result = normalize_plan_artifact(plan)

    assert "day_1:active_session_nonpositive_duration:0" in result.warnings
    assert result.plan["plan"]["days"][0]["duration_minutes"] == before["duration_minutes"]
