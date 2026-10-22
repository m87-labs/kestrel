"""Request-local EOS control preserves normal stopping by default."""
import asyncio
from types import SimpleNamespace

import pytest

from kestrel.engine import InferenceEngine
from kestrel.scheduler import GeneratedPrefix
from kestrel.skills import BuiltRequest


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('settings,expected', [(None, False), ({'ignore_eos': True}, True)])
def test_skill_request_forwards_ignore_eos(stream, settings, expected):
    engine = object.__new__(InferenceEngine)
    built = BuiltRequest(request_context=object(), max_new_tokens=4,
                         temperature=0, top_p=1)
    engine._skill_registry = lambda: SimpleNamespace(resolve=lambda _: SimpleNamespace(
        build_request=lambda *args: built))
    captured = {}

    async def submit(context, **kwargs):
        captured.update(kwargs)
        return context

    engine.submit = submit
    engine.submit_streaming = submit
    result = asyncio.run(engine._run_skill(
        'chat', image=None, prompt={}, settings=settings, stream=stream))
    assert result is built.request_context
    assert captured['ignore_eos'] is expected


@pytest.mark.parametrize('value', [None, 0, 1, 'false', []])
def test_direct_request_rejects_non_boolean_ignore_eos_before_startup(value):
    engine = object.__new__(InferenceEngine)
    engine._shutdown = False
    with pytest.raises(TypeError, match='ignore_eos must be a boolean'):
        asyncio.run(engine._submit_request(
            max_new_tokens=4, request_context=object(), adapter=None, image=None,
            encoder_input=None, temperature=0, top_p=1, return_logprobs=None,
            generated_prefix=GeneratedPrefix(), suppress_next_token_ids=None,
            stream_queue=None, skill='chat', ignore_eos=value))
