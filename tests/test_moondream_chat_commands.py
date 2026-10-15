"""Native chat commands use real skill states and scheduler stream counters."""

import json
from types import SimpleNamespace

import pytest

from kestrel.models.moondream.skills.caption import CaptionSkill
from kestrel.models.moondream.skills.chat import MoondreamChatSkill
from kestrel.models.moondream.skills.detect import DetectSkill
from kestrel.models.moondream.skills.point import PointSkill
from kestrel.runtime.tokens import CoordToken, SizeToken, TextToken
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.types import GeneratedPrefix, GenerationRequest, RequestLifecycle
from kestrel.skills.base import DecodeStep
from kestrel.skills.chat import ChatRequest
from tests.test_chat_skill import _IMG, _md_runtime


SKILLS = {"detect": DetectSkill(), "point": PointSkill(), "caption": CaptionSkill()}


def _messages(text, count=1):
    return [
        {
            "role": "user",
            "content": [
                *[
                    {"type": "image_url", "image_url": {"url": _IMG}}
                    for _ in range(count)
                ],
                {"type": "text", "text": text},
            ],
        }
    ]


def _build(text, count=1, *, stream=False, messages=None, settings=None):
    return MoondreamChatSkill().build_request(
        None,
        {
            "messages": messages if messages is not None else _messages(text, count),
            "reasoning": True,
            "stream": stream,
        },
        settings,
    )


def _runtime():
    runtime = _md_runtime("moondream3-preview")
    runtime.tokenizer.decode = lambda ids: "".join(chr(i) for i in ids if i != 0)
    runtime.prompt_template.detect = lambda: SimpleNamespace(prefix=[51], suffix=[52])
    runtime.prompt_template.point = lambda: SimpleNamespace(prefix=[61], suffix=[62])
    runtime.prompt_template.caption = lambda length: [
        71,
        {"short": 72, "normal": 73, "long": 74}[length],
    ]
    return runtime


@pytest.mark.parametrize(
    "operation,arguments,tokens,expected",
    [
        (
            "detect",
            {"object": "person"},
            [CoordToken(0.25), CoordToken(0.75), SizeToken(0.125, 0.25)],
            {
                "objects": [
                    {"x_min": 0.1875, "y_min": 0.625, "x_max": 0.3125, "y_max": 0.875}
                ]
            },
        ),
        (
            "point",
            {"object": "person"},
            [CoordToken(0.123456789), CoordToken(0.987654321)],
            {"points": [{"x": 0.123456789, "y": 0.987654321}]},
        ),
        (
            "caption",
            {"length": "normal"},
            [TextToken(ord(c)) for c in 'A "person".'],
            {"caption": 'A "person".'},
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", ["stop", "length"])
def test_commands_reuse_native_prompt_decode_schema_and_usage(
    operation, arguments, tokens, expected, stream, reason
):
    chat = MoondreamChatSkill()
    runtime = _runtime()
    built = _build(f"/{operation} {json.dumps(arguments)}", stream=stream)
    native = SKILLS[operation].build_request(b"hi", arguments, None)
    assert built.image == b"hi"  # A single native image, not a chat image tuple.
    assert (built.max_new_tokens, built.temperature, built.top_p) == (
        native.max_new_tokens,
        native.temperature,
        native.top_p,
    )
    prompt = chat.build_prompt_tokens(runtime, built.request_context)
    assert prompt == list(
        SKILLS[operation].build_prompt_tokens(runtime, native.request_context)
    )
    updates = []
    request = GenerationRequest(
        request_id=1,
        prompt="command",
        prompt_tokens=prompt,
        max_new_tokens=built.max_new_tokens,
        skill=chat,
        request_context=built.request_context,
        stream_callback=updates.append if stream else None,
    )
    state = chat.create_state(runtime, request, built.request_context)
    native_state = SKILLS[operation].create_state(
        runtime, request, native.request_context
    )
    assert state.mask_is_stateful == native_state.mask_is_stateful
    state.on_prefill(runtime)
    lifecycle = RequestLifecycle(
        request=request,
        skill_state=state,
        sequence_state=SimpleNamespace(prompt_length=743, reused_page_count=0),
    )
    request.lifecycle = lifecycle
    for index, token in enumerate([*tokens, TextToken(runtime.prompt_template.eos_id)]):
        assert state.allowed_token_ids(runtime) == native_state.allowed_token_ids(
            runtime
        )
        assert state.suppressed_token_ids(runtime) == native_state.suppressed_token_ids(
            runtime
        )
        lifecycle.stage_token(runtime, token)
        native_state.consume_step(runtime, DecodeStep(token=token, position=index))
    assert not any(update.text or update.reasoning for update in updates)
    lifecycle.finish_reason = reason
    # Exercise the real scheduler's finalize -> publish-output drain.
    result = GenerationScheduler._build_result(
        SimpleNamespace(runtime=runtime), lifecycle
    )
    content = result.output["message"]["content"]
    assert json.loads(content) == expected
    assert json.loads(content) == native_state.finalize(runtime, reason=reason).output
    assert result.tokens == [*tokens, TextToken(runtime.prompt_template.eos_id)]
    assert result.output["finish_reason"] == reason
    assert result.metrics.prompt_tokens == 743
    assert result.metrics.decode_tokens == len(tokens) + 1
    if stream:
        published = [u for u in updates if u.text]
        assert len(published) == 1 and published[0].text == content
        assert published[0].input_tokens == 743
        assert published[0].output_tokens == result.metrics.decode_tokens
        assert published[0].token_index is None
        assert not lifecycle.publish_stream_output(runtime)


@pytest.mark.parametrize("operation", ["detect", "point", "caption"])
@pytest.mark.parametrize("count", [0, 2])
def test_command_requires_exactly_one_image(operation, count):
    args = '{"length":"normal"}' if operation == "caption" else '{"object":"person"}'
    with pytest.raises(ValueError, match="requires exactly one image"):
        _build(f"/{operation} {args}", count)


@pytest.mark.parametrize(
    "text,match",
    [
        ("/detect", "JSON object"),
        ("/point nope", "JSON object"),
        ('/caption {"length":', "JSON object"),
        ("/caption []", "JSON object"),
        ('/detect {"object":"person"} trailing', "JSON object"),
        ("/detect {}", "non-empty string"),
        ('/point {"object":null}', "non-empty string"),
        ('/detect {"object":5}', "non-empty string"),
        ('/point {"object":" "}', "non-empty string"),
        ('/caption {"length":9}', "length must be a string"),
        ('/caption {"length":"huge"}', "length must be one of"),
        ('/detect {"object":"person","bogus":true}', "unsupported fields: bogus"),
        ('/point {"object":"person","length":"normal"}', "unsupported fields: length"),
        ('/caption {"object":"person"}', "unsupported fields: object"),
        ('/detect {"object":"person","max_objects":true}', "positive integer"),
        ('/point {"object":"person","max_objects":0}', "positive integer"),
        ('/detect {"object":"person","max_objects":1.2}', "positive integer"),
        ('/point {"object":"person","spatial_refs":true}', "coordinate arrays"),
        ('/point {"object":"person","spatial_refs":[2]}', "coordinate arrays"),
        ('/point {"object":"person","spatial_refs":[[true,0]]}', "coordinate arrays"),
        ('/point {"object":"person","spatial_refs":[[0,2]]}', "normalised"),
        ('/point {"object":"person","spatial_refs":[[0]]}', "2 .* or 4"),
        ('/point {"object":"person","spatial_refs":[[NaN,0]]}', "non-finite"),
    ],
)
def test_invalid_command_arguments(text, match):
    with pytest.raises(ValueError, match=match):
        _build(text)


@pytest.mark.parametrize("operation,budget", [("detect", 7), ("point", 5)])
def test_native_object_limit(operation, budget):
    built = _build(f'/{operation} {{"object":"person","max_objects":2}}')
    assert built.max_new_tokens == budget


def test_point_spatial_refs_and_caption_defaults():
    built = _build('/point {"object":"person","spatial_refs":[[0.25,0.75]]}')
    assert built.request_context.context.spatial_refs == ((0.25, 0.75),)
    assert _build("/caption {}").request_context.context.length == "normal"


@pytest.mark.parametrize(
    "text",
    [
        "Hello",
        'Please /detect {"object":"person"}',
        '/detective {"object":"person"}',
        '/segment {"object":"person"}',
        ' /detect {"object":"person"}',
    ],
)
def test_ordinary_chat_is_unchanged(text):
    built = _build(text)
    assert isinstance(built.request_context, ChatRequest)
    assert built.request_context.messages[-1].text == text


def test_commands_only_in_final_user_text():
    messages = [
        {"role": "system", "content": '/detect {"object":"person"}'},
        *_messages('/point {"object":"person"}'),
        {"role": "assistant", "content": '/caption {"length":"normal"}'},
        {"role": "user", "content": "Continue normally"},
    ]
    assert isinstance(_build("", messages=messages).request_context, ChatRequest)
    # Assistant image data is never a source for a command's image.
    with pytest.raises(ValueError, match="images only in user messages"):
        _build(
            "",
            messages=[
                {"role": "assistant", "content": _messages("ignored")[0]["content"]},
                {"role": "user", "content": "/caption {}"},
            ],
        )


def test_single_image_can_be_in_user_history_but_multiple_images_are_rejected():
    messages = [
        *_messages("Look"),
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "/caption {}"},
    ]
    assert _build("", messages=messages).image == b"hi"
    messages[-1] = _messages("/caption {}")[0]
    with pytest.raises(ValueError, match="exactly one image"):
        _build("", messages=messages)


def test_generated_prefix_not_billed_as_new_output():
    from tests.scheduler.test_request_metrics import _lifecycle

    lifecycle = _lifecycle()
    lifecycle.request.generated_prefix = GeneratedPrefix(tokens=(TextToken(7),))
    lifecycle.skill_state.append_token(TextToken(7))
    updates = []
    lifecycle.request.stream_callback = updates.append
    lifecycle.stage_token(object(), TextToken(8))
    assert updates[0].output_tokens == 1
    assert updates[0].input_tokens == lifecycle.request.prompt_length


def test_caption_output_command_is_returned_as_data():
    runtime = _runtime()
    built = _build("/caption {}", stream=True)
    state = MoondreamChatSkill().create_state(
        runtime, SimpleNamespace(), built.request_context
    )
    caption = '/detect {"object":"person"}'
    for index, char in enumerate(caption):
        state.consume_step(
            runtime, DecodeStep(token=TextToken(ord(char)), position=index)
        )
    result = state.finalize(runtime, reason="stop")
    assert json.loads(result.output["message"]["content"]) == {"caption": caption}
    assert state.pop_stream_output(runtime) == {"text": result.text}
    assert state.pop_stream_output(runtime) == {}


@pytest.mark.parametrize("operation", ["detect", "point"])
def test_empty_native_results_remain_json(operation):
    built = _build(f'/{operation} {{"object":"person"}}')
    state = MoondreamChatSkill().create_state(
        _runtime(), SimpleNamespace(), built.request_context
    )
    key = "objects" if operation == "detect" else "points"
    assert json.loads(state.finalize(_runtime(), reason="stop").text) == {key: []}
