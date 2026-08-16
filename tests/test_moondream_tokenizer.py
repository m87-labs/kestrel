from tokenizers import Tokenizer
from tokenizers.models import WordLevel

import kestrel.runtime.tokenizer as tokenizer_module
from kestrel.runtime.tokenizer import load_tokenizer


def _save_test_tokenizer(path) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.save(str(path))


def test_load_tokenizer_from_local_json_ignores_revision(tmp_path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _save_test_tokenizer(tokenizer_path)

    tokenizer = load_tokenizer(
        "unused/repo", tokenizer_path, revision="remote-only-revision"
    )

    assert tokenizer.encode("hello").ids == [1]


def test_load_tokenizer_from_directory(tmp_path) -> None:
    _save_test_tokenizer(tmp_path / "tokenizer.json")

    tokenizer = load_tokenizer("unused/repo", tmp_path)

    assert tokenizer.encode("hello").ids == [1]


def test_load_remote_tokenizer_uses_declared_revision(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    sentinel = object()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(identifier: str, *, revision: str):
            calls.append((identifier, revision))
            return sentinel

    monkeypatch.setattr(tokenizer_module, "Tokenizer", FakeTokenizer)

    assert load_tokenizer(
        "owner/tokenizer", None, revision="immutable-commit"
    ) is sentinel
    assert calls == [("owner/tokenizer", "immutable-commit")]
