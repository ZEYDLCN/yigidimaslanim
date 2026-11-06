"""Readiness needs evidence: a recorded, passing, current probe for this configuration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from notes_graphrag.api.dependencies import build_context
from notes_graphrag.api.routes.health import _probe_check
from notes_graphrag.core.config import Settings
from notes_graphrag.providers.capability_probes import profile_fingerprint

NOW = datetime.now(UTC)


def _ctx(**over):
    base = dict(
        _env_file=None,
        extraction_model="chat-m",
        answer_model="chat-m",
        embedding_model="emb-m",
        embedding_dimensions=1024,
        embedding_profile="qwen-v1",
    )
    base.update(over)
    return build_context(Settings(**base), demo=False, backend="memory")


def test_production_without_any_probe_is_not_ready():
    ctx = _ctx()
    ok, detail = _probe_check(ctx)
    assert ok is False and "no capability probe" in detail


def test_current_passing_probe_makes_it_ready():
    ctx = _ctx()
    ctx.store.record_probe(
        profile_fingerprint(ctx.settings), True, [{"name": "x", "ok": True}], NOW
    )
    assert _probe_check(ctx)[0] is True


@pytest.mark.parametrize("case", ["failed", "other_config", "stale"])
def test_probe_that_does_not_prove_readiness_is_rejected(case):
    ctx = _ctx()
    fp = profile_fingerprint(ctx.settings)
    if case == "failed":
        ctx.store.record_probe(fp, False, [{"name": "x", "ok": False}], NOW)
    elif case == "other_config":
        ctx.store.record_probe("0" * 24, True, [], NOW)
    else:
        ctx.store.record_probe(fp, True, [], NOW - timedelta(hours=25))
    assert _probe_check(ctx)[0] is False


def test_latest_probe_wins_over_older_failures():
    ctx = _ctx()
    fp = profile_fingerprint(ctx.settings)
    ctx.store.record_probe(fp, False, [], NOW - timedelta(minutes=10))
    ctx.store.record_probe(fp, True, [], NOW)
    assert _probe_check(ctx)[0] is True


def test_fingerprint_changes_with_model_dimension_or_profile():
    base = profile_fingerprint(_ctx().settings)
    assert profile_fingerprint(_ctx(embedding_dimensions=768).settings) != base
    assert profile_fingerprint(_ctx(embedding_profile="other").settings) != base
    assert profile_fingerprint(_ctx(answer_model="other").settings) != base


def test_demo_mode_is_exempt_and_says_so():
    demo = build_context(Settings(_env_file=None), demo=True, backend="memory")
    ok, detail = _probe_check(demo)
    assert ok is True and "demo" in detail
