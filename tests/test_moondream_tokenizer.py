from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from kestrel.models.moondream.tokenizer import load_tokenizer


def _save_test_tokenizer(path) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.save(str(path))


def test_load_tokenizer_from_local_json(tmp_path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _save_test_tokenizer(tokenizer_path)

    tokenizer = load_tokenizer("unused/repo", tokenizer_path)

    assert tokenizer.encode("hello").ids == [1]


def test_load_tokenizer_from_directory(tmp_path) -> None:
    _save_test_tokenizer(tmp_path / "tokenizer.json")

    tokenizer = load_tokenizer("unused/repo", tmp_path)

    assert tokenizer.encode("hello").ids == [1]
