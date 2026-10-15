# Moondream native commands through chat

Moondream's chat skill recognizes `/detect`, `/point`, and `/caption` at the
start of the final user message's text. For example, send an ordinary
`/chat/completions` request with one embedded image and this user text:

```text
/detect {"object":"person"}
/point {"object":"person"}
/caption {"length":"normal"}
```

Use one command per request. Its arguments must be one JSON object with no
trailing text or unsupported fields:

| Command | Arguments |
| --- | --- |
| `/detect` | Required nonempty string `object`; optional positive integer `max_objects` |
| `/point` | Required nonempty string `object`; optional positive integer `max_objects` and native `spatial_refs` |
| `/caption` | Optional `length`: `short`, `normal` (default), or `long` |

`spatial_refs` retains the native point operation's normalized point/bounding-box
validation. Sampling settings stay in the ordinary request settings. Detect and
point retain their native token budgets, including budgets derived from
`max_objects`.

Commands require exactly one image across the conversation, supplied in a user
message. A single image in an earlier user turn is accepted. Conversation text
is not sent to the native operation; its prompt comes from the JSON arguments.
Only the final user's text is inspected. Earlier turns, model output, image
bytes, and ordinary chat text do not trigger operations. Segmentation is not
exposed as a command.

The assistant's `content` is a JSON string containing the unchanged native
result: `{"objects":[...]}`, `{"points":[...]}`, or `{"caption":"..."}`. Coordinates
retain native normalization and precision. Streaming buffers generation and
publishes this complete string in one content update. The engine's cumulative
usage counts native generated tokens, including coordinate/size/EOS tokens,
rather than tokens in the serialized JSON. FAL adapts that update into its
existing SSE content, finish, usage, and `[DONE]` frames. Prime forwards the
ordinary chat request and needs no command-specific route or code.

CPU regression tests (no model weights required):

```bash
python -m pytest tests/test_moondream_chat_commands.py tests/test_chat_skill.py tests/scheduler/test_request_metrics.py -q
```

Before deployment, compare all three commands against their native operations
on the same image and settings for both Moondream 3 Preview and Moondream 3.1.
Check both normal and streaming requests through FAL, then through Prime.
