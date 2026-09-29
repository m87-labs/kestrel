"""Every published Parakeet id is registered, and each resolves at the revision it should.

The upstream checkpoint is pinned: that repository is not ours and its weights could change under a release.
Ours are not, because a commit hash stops resolving the moment a repository's history is rewritten.
"""

import pytest

from kestrel.models import get_spec
from kestrel.models.parakeet_tdt import (
    MODEL_ID,
    REVISION,
    TERNARY_MODEL_ID,
    ULTRA_MODEL_ID,
    load_parakeet_tdt,
)
import kestrel.models.parakeet_tdt.weights as weights

OURS = (TERNARY_MODEL_ID, ULTRA_MODEL_ID)


@pytest.mark.parametrize("model_id", (MODEL_ID,) + OURS)
def test_every_parakeet_id_is_registered(model_id) -> None:
    assert get_spec(model_id).repo_id == model_id


def test_only_the_upstream_checkpoint_is_pinned() -> None:
    assert get_spec(MODEL_ID).revision == REVISION
    assert [get_spec(model_id).revision for model_id in OURS] == [None, None]


@pytest.mark.parametrize(
    "model_id, revision, wants_manifest",
    [
        (MODEL_ID, REVISION, False),
        (TERNARY_MODEL_ID, "main", True),
        (ULTRA_MODEL_ID, "main", False),
    ],
)
def test_the_loader_asks_for_that_same_revision(monkeypatch, model_id, revision, wants_manifest) -> None:
    seen = {}

    def resolve(checkpoint, *, revision, filenames, local_files_only):
        seen.update(checkpoint=str(checkpoint), revision=revision, filenames=tuple(filenames))
        raise RuntimeError("stop after resolving")

    monkeypatch.setattr(weights, "resolve_checkpoint", resolve)
    with pytest.raises(RuntimeError, match="stop after resolving"):
        load_parakeet_tdt(model_id)
    assert seen["checkpoint"] == model_id
    assert seen["revision"] == revision
    assert ("ternary.json" in seen["filenames"]) is wants_manifest


def test_an_unregistered_repository_is_read_at_its_main(monkeypatch) -> None:
    """The fallback is `main`, not the upstream pin: that hash names one repository and no other."""
    seen = {}

    def resolve(checkpoint, *, revision, filenames, local_files_only):
        seen.update(revision=revision)
        raise RuntimeError("stop after resolving")

    monkeypatch.setattr(weights, "resolve_checkpoint", resolve)
    with pytest.raises(RuntimeError, match="stop after resolving"):
        load_parakeet_tdt("someone/a-parakeet-of-their-own")
    assert seen["revision"] == "main"


def test_an_explicit_revision_still_wins(monkeypatch) -> None:
    seen = {}

    def resolve(checkpoint, *, revision, filenames, local_files_only):
        seen.update(revision=revision)
        raise RuntimeError("stop after resolving")

    monkeypatch.setattr(weights, "resolve_checkpoint", resolve)
    with pytest.raises(RuntimeError, match="stop after resolving"):
        load_parakeet_tdt(ULTRA_MODEL_ID, revision="kestrel-pin-0.8.1")
    assert seen["revision"] == "kestrel-pin-0.8.1"
