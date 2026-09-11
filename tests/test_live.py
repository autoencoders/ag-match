"""Live smoke test against a real model. Set AG_MATCH_LIVE=1 and GOOGLE_API_KEY to run."""

from __future__ import annotations

import os

import pytest

from ag_match import Matcher

pytestmark = pytest.mark.skipif(
    not os.environ.get("AG_MATCH_LIVE"), reason="set AG_MATCH_LIVE=1 to run live tests"
)


async def test_misspelled_name_matches(tool):
    run = await Matcher(tool, model=os.environ.get("AG_MATCH_MODEL")).match_async(
        "Acmee Holdngs Intl.", context={"country": "US"}
    )
    print(run.model_dump_json(indent=2, exclude={"messages"}))
    assert run.decision.status == "matched"
    assert run.match.id == "c1"
    assert run.usage.tool_calls <= 4


async def test_absent_name_is_no_match(tool):
    run = await Matcher(tool, model=os.environ.get("AG_MATCH_MODEL")).match_async(
        "Vandelay Industries"
    )
    print(run.model_dump_json(indent=2, exclude={"messages"}))
    assert run.decision.status == "no_match"
