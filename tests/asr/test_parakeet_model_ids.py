"""Every published Parakeet id is registered and loads at its pinned revision."""

import pytest

from kestrel.models import get_spec
from kestrel.models.parakeet_tdt import (
    MODEL_ID,
    REVISION,
    TERNARY_MODEL_ID,
    TERNARY_REVISION,
    ULTRA_MODEL_ID,
    ULTRA_REVISION,
    load_parakeet_tdt,
)
import kestrel.models.parakeet_tdt.weights as weights


@pytest.mark.parametrize(
    "model_id, revision",
    [(MODEL_ID, REVISION), (TERNARY_MODEL_ID, TERNARY_REVISION), (ULTRA_MODEL_ID, ULTRA_REVISION)],
)
def test_parakeet_ids_are_registered_at_their_pinned_revision(model_id, revision) -> None:
    spec = get_spec(model_id)
    assert spec.repo_id == model_id
    assert spec.revision == revision


@pytest.mark.parametrize(
    "model_id, revision, wants_manifest",
    [
        (MODEL_ID, REVISION, False),
        (TERNARY_MODEL_ID, TERNARY_REVISION, True),
        (ULTRA_MODEL_ID, ULTRA_REVISION, False),
    ],
)
def test_loader_resolves_each_id_at_its_pin(monkeypatch, model_id, revision, wants_manifest) -> None:
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
