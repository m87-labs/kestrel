"""Tests for prefix cache token implementations."""

import pytest

from kestrel.prefix_cache import CacheToken, ImageToken


class TestImageToken:
    """Tests for ImageToken CacheToken implementation."""

    def test_cache_key_discriminates_images(self) -> None:
        """Different image hashes produce different cache keys."""
        token1 = ImageToken(content_hash=12345, kv_length_=729)
        token2 = ImageToken(content_hash=67890, kv_length_=729)

        assert token1.cache_key() != token2.cache_key()

    def test_cache_key_format(self) -> None:
        """Cache key uses type discriminator 3 and includes kv_length."""
        token = ImageToken(content_hash=42, kv_length_=729)
        key = token.cache_key()

        assert key == (3, 42, 729)
        assert key[0] == 3  # Type discriminator for image tokens
        assert key[2] == 729  # kv_length for future-proofing

    def test_kv_length(self) -> None:
        """kv_length returns the stored value."""
        token = ImageToken(content_hash=1, kv_length_=729)
        assert token.kv_length() == 729

        token2 = ImageToken(content_hash=1, kv_length_=1024)
        assert token2.kv_length() == 1024

    def test_implements_cache_token_protocol(self) -> None:
        """ImageToken implements CacheToken protocol."""
        token = ImageToken(content_hash=1, kv_length_=729)

        assert isinstance(token, CacheToken)
        assert hasattr(token, "cache_key")
        assert hasattr(token, "kv_length")
        assert callable(token.cache_key)
        assert callable(token.kv_length)

    def test_frozen_dataclass(self) -> None:
        """ImageToken is immutable."""
        token = ImageToken(content_hash=1, kv_length_=729)

        with pytest.raises(AttributeError):
            token.content_hash = 2  # type: ignore

    def test_equality(self) -> None:
        """Tokens with same values are equal."""
        token1 = ImageToken(content_hash=42, kv_length_=729)
        token2 = ImageToken(content_hash=42, kv_length_=729)
        token3 = ImageToken(content_hash=42, kv_length_=100)

        assert token1 == token2
        assert token1 != token3


class TestRuntimeTokens:
    """Tests for runtime Token CacheToken implementations."""

    def test_text_token_cache_key(self) -> None:
        """TextToken uses type discriminator 0."""
        from kestrel.moondream.runtime import TextToken

        token = TextToken(token_id=42)
        key = token.cache_key()

        assert key == (0, 42)
        assert key[0] == 0  # Type discriminator for text tokens

    def test_text_token_kv_length(self) -> None:
        """TextToken occupies 1 KV position."""
        from kestrel.moondream.runtime import TextToken

        token = TextToken(token_id=42)
        assert token.kv_length() == 1

    def test_text_token_implements_cache_token(self) -> None:
        """TextToken implements CacheToken protocol."""
        from kestrel.moondream.runtime import TextToken

        token = TextToken(token_id=42)
        assert isinstance(token, CacheToken)

    def test_coord_token_cache_key(self) -> None:
        """CoordToken uses type discriminator 1."""
        from kestrel.moondream.runtime import CoordToken

        token = CoordToken(pos=0.5)
        key = token.cache_key()

        assert key == (1, 0.5)
        assert key[0] == 1  # Type discriminator for coord tokens

    def test_coord_token_kv_length(self) -> None:
        """CoordToken occupies 1 KV position."""
        from kestrel.moondream.runtime import CoordToken

        token = CoordToken(pos=0.5)
        assert token.kv_length() == 1

    def test_size_token_cache_key(self) -> None:
        """SizeToken uses type discriminator 2."""
        from kestrel.moondream.runtime import SizeToken

        token = SizeToken(width=0.3, height=0.4)
        key = token.cache_key()

        assert key == (2, 0.3, 0.4)
        assert key[0] == 2  # Type discriminator for size tokens

    def test_size_token_kv_length(self) -> None:
        """SizeToken occupies 1 KV position."""
        from kestrel.moondream.runtime import SizeToken

        token = SizeToken(width=0.3, height=0.4)
        assert token.kv_length() == 1

    def test_all_tokens_have_distinct_discriminators(self) -> None:
        """All token types use distinct type discriminators."""
        from kestrel.moondream.runtime import CoordToken, SizeToken, TextToken

        text_key = TextToken(token_id=1).cache_key()
        coord_key = CoordToken(pos=0.5).cache_key()
        size_key = SizeToken(width=0.1, height=0.2).cache_key()
        image_key = ImageToken(content_hash=123, kv_length_=729).cache_key()

        discriminators = {text_key[0], coord_key[0], size_key[0], image_key[0]}
        assert len(discriminators) == 4  # All distinct


class TestTokenCacheKeyUniqueness:
    """Tests for cache key uniqueness across token types."""

    def test_different_types_never_collide(self) -> None:
        """Different token types never produce the same cache key."""
        from kestrel.moondream.runtime import CoordToken, SizeToken, TextToken

        # Same "value" but different types
        text_token = TextToken(token_id=1)
        coord_token = CoordToken(pos=1.0)
        size_token = SizeToken(width=1.0, height=1.0)
        image_token = ImageToken(content_hash=1, kv_length_=1)

        keys = {
            text_token.cache_key(),
            coord_token.cache_key(),
            size_token.cache_key(),
            image_token.cache_key(),
        }

        assert len(keys) == 4  # All distinct

    def test_text_tokens_with_same_id_match(self) -> None:
        """Text tokens with the same ID have the same cache key."""
        from kestrel.moondream.runtime import TextToken

        token1 = TextToken(token_id=42)
        token2 = TextToken(token_id=42)

        assert token1.cache_key() == token2.cache_key()
