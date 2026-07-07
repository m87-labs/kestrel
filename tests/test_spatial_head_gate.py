"""The moondream spatial-head gate (`SkillState.emits_spatial_tokens`).

The runtime's ``post_sample`` skips the coord/size decode head (a ~12.6MB matvec
plus the coord/size sample launches) whenever no sequence in the batch is in a
spatial-emitting phase. That decision is driven entirely by
``SkillState.emits_spatial_tokens``; these tests pin the per-skill / per-phase
answer so a future skill can't silently regress the gate (either dropping the
head where a coord/size value is consumed, or keeping it where it is pure waste).
"""

from __future__ import annotations

from kestrel.models.moondream.skills.caption import CaptionSkillState
from kestrel.models.moondream.skills.detect import DetectSkillState
from kestrel.models.moondream.skills.point import PointSkillState
from kestrel.models.moondream.skills.query import QuerySkillState
from kestrel.models.moondream.skills.segment import SegmentSkillState
from kestrel.skills.base import SkillState


def _bare(cls):
    # ``emits_spatial_tokens`` never touches construction-time state (Query only
    # reads ``_collecting_reasoning``, set explicitly below), so bypass __init__.
    return object.__new__(cls)


def test_spatial_skills_keep_the_head() -> None:
    # Point (coord/coord), detect (coord/coord/size) and segment (interleaved
    # text + bbox/polygon) consume a spatial value throughout their run.
    for cls in (PointSkillState, DetectSkillState, SegmentSkillState):
        assert _bare(cls).emits_spatial_tokens is True


def test_text_skills_drop_the_head() -> None:
    # The base contract and a caption are pure text -- no coord/size consumed.
    assert _bare(SkillState).emits_spatial_tokens is False
    assert _bare(CaptionSkillState).emits_spatial_tokens is False


def test_query_head_tracks_reasoning_phase() -> None:
    # A query only consumes coord tokens while collecting grounded reasoning;
    # its answer phase (and every non-reasoning query) is pure text.
    st = _bare(QuerySkillState)
    st._collecting_reasoning = True
    assert st.emits_spatial_tokens is True
    st._collecting_reasoning = False
    assert st.emits_spatial_tokens is False
