"""Focused ownership checks for asynchronous Qwen3-TTS codec output."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from kestrel.runtime.tokens import TextToken
from kestrel.skills.base import DecodeStep
from kestrel.models.qwen3_tts.codec import IncrementalCodecState
from kestrel.models.qwen3_tts.config import SAMPLE_RATE, SAMPLES_PER_FRAME
from kestrel.models.qwen3_tts.runtime import (
    _CodecBuffer,
    _CodecJob,
    _CodecLane,
    Qwen3TTSRuntime,
)
from kestrel.models.qwen3_tts.skill import Qwen3TTSSynthesisState


class _Event:
    def __init__(
        self,
        predecessors: tuple["_Event", ...] = (),
    ) -> None:
        self.complete = False
        self.predecessors = predecessors
        self.synchronize_calls = 0

    def query(self) -> bool:
        return self.complete

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.complete = True
        for predecessor in self.predecessors:
            predecessor.complete = True

def _job(
    state: Qwen3TTSSynthesisState,
    event: _Event,
    value: float,
) -> _CodecJob:
    pcm = torch.full((1, SAMPLES_PER_FRAME), value, dtype=torch.float32)
    buffer = _CodecBuffer(
        pcm_cpu=pcm,
        copy_done=event,
    )
    return _CodecJob(
        buffer=buffer,
        states=(state,),
        frame_count=1,
        skip_samples=(0,),
    )


def _stream_state() -> Qwen3TTSSynthesisState:
    state = object.__new__(Qwen3TTSSynthesisState)
    state.synthesis = SimpleNamespace(stream=True)
    state.request = SimpleNamespace(submitted_at=99.0)
    state._pcm = []
    state._streamed_chunks = 0
    state._playback_started_at = None
    state._streamed_samples = 0
    state._onset = SimpleNamespace(push=lambda chunk: chunk)
    return state


def _capture_codec_dispatches(runtime: Qwen3TTSRuntime) -> list[tuple]:
    decoded = []

    def decode(states, lanes, frame_count, **_kwargs):
        decoded.append((tuple(states), frame_count))
        for lane in lanes:
            lane.buffered_frames -= frame_count
            lane.chunk_index += 1

    runtime._decode_codec_group = decode
    return decoded


def test_terminal_codec_drain_streams_all_completed_jobs(monkeypatch) -> None:
    monkeypatch.setattr("kestrel.models.qwen3_tts.skill.time.perf_counter", lambda: 100.0)
    first_event = _Event()
    second_event = _Event((first_event,))
    state = _stream_state()
    assert state.next_output_deadline(None) == 99.0
    first = _job(state, first_event, 1.0)
    second = _job(state, second_event, 2.0)
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime._pending_codec_jobs = deque((first, second))
    runtime._free_codec_buffers = []
    runtime._unpublished_codec_states = set()
    runtime._codec_lanes = {state: _CodecLane(0, IncrementalCodecState(), False)}

    runtime._poll_codec_jobs()
    assert state._pcm == []
    assert first_event.synchronize_calls == 0

    runtime._drain_codec_states((state,))
    output = state.pop_stream_output(None)

    assert output is not None
    assert output["sample_rate"] == SAMPLE_RATE
    audio = output["audio"]
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (2 * SAMPLES_PER_FRAME,)
    assert np.all(audio[:SAMPLES_PER_FRAME] == 1.0)
    assert np.all(audio[SAMPLES_PER_FRAME:] == 2.0)
    assert state.pop_stream_output(None) is None
    assert state.next_output_deadline(None) == (
        100.0 + (audio.size - 2 * SAMPLES_PER_FRAME) / SAMPLE_RATE
    )
    state.synthesis.stream = False
    assert state.next_output_deadline(None) is None
    assert second_event.synchronize_calls == 1
    assert list(runtime._pending_codec_jobs) == []
    assert runtime._free_codec_buffers[0] is first.buffer
    assert runtime._free_codec_buffers[1] is second.buffer


def test_completed_codec_job_retains_notification_until_callback_available() -> None:
    state = _stream_state()
    event = _Event()
    event.complete = True
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime._pending_codec_jobs = deque((_job(state, event, 1.0),))
    runtime._free_codec_buffers = []
    runtime._unpublished_codec_states = set()
    runtime._codec_lanes = {state: _CodecLane(0, IncrementalCodecState(), False)}
    ready = []

    runtime._poll_codec_jobs()
    assert len(state._pcm) == 1

    runtime._poll_codec_jobs(stream_output_ready=ready.append)

    assert ready == [state]
    assert runtime._unpublished_codec_states == set()


def test_codec_ramp_waits_for_audible_reserve() -> None:
    state = _stream_state()
    state._onset = SimpleNamespace(push=lambda chunk: chunk if chunk.any() else chunk[:0])
    runtime = object.__new__(Qwen3TTSRuntime)
    lane = _CodecLane(0, IncrementalCodecState(), False)
    runtime._codec_lanes = {state: lane}
    runtime._free_codec_buffers = []
    runtime._unpublished_codec_states = set()
    # Silent chunks and the first audible chunk both need a short refill.
    for value, expected in ((0.0, 1), (1.0, 1), (1.0, 3)):
        lane.chunk_index = 3
        runtime._finish_codec_job(_job(state, _Event(), value))
        assert lane.chunk_index == expected


def test_terminal_token_streams_pcm_buffered_by_the_onset_gate() -> None:
    state = _stream_state()
    state._tokens = []
    state.request.max_new_tokens = 10
    pending = np.array([0.0, 0.1], dtype=np.float32)
    state._onset = SimpleNamespace(finish=lambda: pending)

    state.consume_step(
        SimpleNamespace(eos_token_ids=(99,)),
        DecodeStep(token=TextToken(99), position=0),
    )

    output = state.pop_stream_output(None)
    assert output is not None
    np.testing.assert_array_equal(output["audio"], pending)


def test_cancellation_keeps_committed_frames_and_releases_the_lane() -> None:
    state = object.__new__(Qwen3TTSSynthesisState)
    lane = _CodecLane(3, IncrementalCodecState(), False, buffered_frames=1)
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime._compute_stream = None
    runtime.eos_token_ids = (99,)
    runtime._codec_frames = torch.zeros((4, 16, 12), dtype=torch.int64)
    runtime._codec_frames[3, :, 0] = 7
    runtime._codec_lanes = {state: lane}
    runtime._unpublished_codec_states = {state}
    decoded = []
    runtime._prepared_synthesis = {3: object()}
    runtime._text_continuations = {3: object()}
    runtime._poll_codec_jobs = lambda **_kwargs: None

    def decode(states, lanes, frame_count, **_kwargs):
        decoded.append(
            (states, lanes, frame_count, runtime._codec_frames[3, :, 0].clone())
        )
        for item in lanes:
            item.buffered_frames -= frame_count

    runtime._decode_codec_group = decode
    runtime._drain_codec_states = lambda _states: None
    sequence = SimpleNamespace(
        state=SimpleNamespace(batch_idx=3),
        skill_state=state,
        finalized=True,
    )
    slot = SimpleNamespace(
        scratch={"predictor_frames": torch.full((1, 16), 9, dtype=torch.int64)}
    )

    runtime._materialize_tokens(
        torch.tensor([4]),
        (sequence,),
        torch.tensor([3]),
        (slot, 1),
    )
    runtime._release_runtime_state(3)

    assert [item[2] for item in decoded] == [1]
    assert torch.equal(decoded[0][3], torch.full((16,), 7))
    assert state not in runtime._codec_lanes
    assert state not in runtime._unpublished_codec_states
    assert 3 not in runtime._prepared_synthesis
    assert 3 not in runtime._text_continuations


def test_first_codec_chunk_runs_through_auxiliary_admission() -> None:
    state = _stream_state()
    state._tokens = []
    state.request.max_new_tokens = 10
    lane = _CodecLane(3, IncrementalCodecState(), False, buffered_frames=2)
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime._compute_stream = None
    runtime.eos_token_ids = (99,)
    runtime.max_batch_size = 4
    runtime._codec_frames = torch.zeros((4, 16, 12), dtype=torch.int64)
    runtime._codec_lanes = {state: lane}
    runtime._poll_codec_jobs = lambda **_kwargs: None
    decoded = _capture_codec_dispatches(runtime)
    sequence = SimpleNamespace(
        state=SimpleNamespace(batch_idx=3),
        skill_state=state,
        finalized=False,
    )
    slot = SimpleNamespace(
        scratch={"predictor_frames": torch.full((1, 16), 9, dtype=torch.int64)},
        meta=SimpleNamespace(
            batch_idx=SimpleNamespace(gpu=torch.tensor([3], dtype=torch.int64))
        ),
    )

    runtime._materialize_tokens(
        torch.tensor([4]),
        (sequence,),
        torch.tensor([3]),
        (slot, 1),
    )

    assert decoded == []
    assert lane.buffered_frames == 3
    assert runtime._advance_auxiliary(
        force=False, stream_output_ready=lambda _state: False
    ) is True
    assert decoded == [((state,), 3)]


def test_warm_codec_rows_run_eagerly_in_deadline_order() -> None:
    states = (_stream_state(), _stream_state(), _stream_state())
    for state in states:
        state._playback_started_at = 100.0
    states[0]._streamed_samples = SAMPLE_RATE
    states[1]._streamed_samples = SAMPLE_RATE // 4
    states[2]._streamed_samples = SAMPLE_RATE // 2
    lanes = {
        state: _CodecLane(
            row=index,
            decoder_state=IncrementalCodecState(frame_position=1),
            suppress_bootstrap=False,
            buffered_frames=12,
            chunk_index=3,
        )
        for index, state in enumerate(states)
    }
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime.max_batch_size = 2
    runtime._codec_lanes = lanes
    runtime._poll_codec_jobs = lambda **_kwargs: None
    decoded = _capture_codec_dispatches(runtime)

    assert runtime._advance_auxiliary(
        force=False, stream_output_ready=lambda _state: False
    ) is True
    assert decoded == [((states[1], states[2]), 12)]

    assert runtime._advance_auxiliary(
        force=True, stream_output_ready=lambda _state: False
    ) is True
    assert decoded == [((states[1], states[2]), 12), ((states[0],), 12)]
    assert runtime._can_dispatch_skill(states[0], inflight_steps=0) is True
