"""Descriptor-owned rank-team generated-decode contracts."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from kestrel.runtime.generated_decode import (
    GeneratedDecode,
    GeneratedDecodeTeamMember,
    _program_matches_team,
)
from kestrel.runtime.generated_decode_team import (
    GeneratedDecodeRankTeam,
    _collective_pairs,
    generated_decode_team_members,
)


def test_load_time_weight_selector_matches_late_team_selector(monkeypatch):
    from kestrel.runtime import generated_decode as generated_module

    distributed = SimpleNamespace(descriptor=_descriptor())
    single_descriptor = _descriptor()
    single_descriptor["device_program"]["distribution_plan"] = {}
    single = SimpleNamespace(descriptor=single_descriptor)
    runtime = SimpleNamespace(
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
        max_batch_size=1,
    )
    generator = SimpleNamespace(
        resolve_compatible_programs=lambda *_args, **_kwargs: (
            distributed, single
        )
    )
    monkeypatch.setattr(
        generated_module, "_generated_weight_runtime",
        lambda **_kwargs: generator,
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _device: SimpleNamespace(major=10, minor=0, multi_processor_count=148),
    )
    monkeypatch.setattr(
        generated_module, "_select_program",
        lambda programs, *_args, **_kwargs: (0, programs[0]) if programs else None,
    )
    monkeypatch.setattr(
        generated_module, "_selectable_programs",
        lambda programs, *_args, **_kwargs: tuple(programs),
    )
    kwargs = dict(
        label="Qwen", layer_prefix="layers", required_batch_sizes=(1,),
        required=True,
    )
    assert generated_module.generated_weight_programs_for_loading(
        runtime, torch.nn.Module(), **kwargs
    ) == (single,)
    member = generated_decode_team_members(range(8), topology="route_slot_ep")[0]
    assert generated_module.generated_weight_programs_for_loading(
        runtime, torch.nn.Module(), team_member=member, **kwargs
    ) == (distributed,)


def _descriptor():
    buffers = []
    arguments = []
    for stem, dtype, shape in (
        ("payload", "bf16", [2, 8, 16]),
        ("epoch", "int32", [2, 1]),
    ):
        for view in ("local", "multicast"):
            name = f"team_{stem}_{view}"
            buffers.append({
                "name": name,
                "role": f"{stem}_{view}",
                "dtype": dtype,
                "shape": shape,
                "access": "read_write" if view == "local" else "read",
                "assumed_align": 16,
            })
            arguments.append({
                "name": name,
                "source": "collective",
                "transport": "raw_pointer",
                "dtype": dtype,
                "shape": shape,
                "access": "read_write" if view == "local" else "read",
                "assumed_align": 16,
            })
    return {
        "program": {"program": "rank-team-test"},
        "weights": [],
        "device_program": {
            "distribution_plan": {
                "world_size": 8,
                "strategy": "tensor_parallel",
                "routed_expert_topology": "route_slot_ep",
                "routed_down_collectives": [{"buffers": buffers}],
            },
            "argument_plan": {"arguments": arguments},
        },
    }


def test_generated_decode_team_members_filter_compiler_programs():
    members = generated_decode_team_members(range(8), topology="route_slot_ep")
    assert members == tuple(
        GeneratedDecodeTeamMember(tuple(range(8)), rank, "route_slot_ep")
        for rank in range(8)
    )
    program = SimpleNamespace(descriptor=_descriptor())
    assert _program_matches_team(program, members[0])
    assert not _program_matches_team(program, None)
    assert not _program_matches_team(
        program, GeneratedDecodeTeamMember(tuple(range(8)), 0, "feature_sharded")
    )


def test_team_member_cannot_retain_mutable_device_membership():
    devices = list(range(8))
    member = GeneratedDecodeTeamMember(devices, 0, "route_slot_ep")
    devices[0] = 9
    assert member.devices == tuple(range(8))


def test_distributed_decode_rejects_direct_rank_launch():
    decode = object.__new__(GeneratedDecode)
    decode._spec = SimpleNamespace(team_member=GeneratedDecodeTeamMember(
        tuple(range(8)), 0, "route_slot_ep",
    ))
    decode._team_bound = True
    with pytest.raises(RuntimeError, match="requires a rank team"):
        decode.run(None, 1)
    with pytest.raises(RuntimeError, match="requires a rank team"):
        decode.static_launcher(None, 1)


def test_team_slot_launch_preserves_bound_stream_and_preparation_order():
    decode = object.__new__(GeneratedDecode)
    stream = object()
    slot = SimpleNamespace(slot_id=0, compute_stream=stream)
    invocation = SimpleNamespace(stream=stream)
    decode._spec = SimpleNamespace(
        team_member=GeneratedDecodeTeamMember(tuple(range(8)), 0, "route_slot_ep"),
        bindings=SimpleNamespace(launch_extents=lambda _slot, _batch: {"kv_len": 2}),
        label="Qwen",
        preparation_callbacks={},
    )
    decode._team_bound = True
    decode._program_for = lambda _batch, _extents: (0, object())
    decode._slots = {(0, 0): SimpleNamespace(
        invocation=invocation,
        scalar_names=frozenset({"kv_len"}),
        required_launch_extents=frozenset({"kv_len"}),
    )}
    assert decode.team_slot_launch(slot, 1) == (invocation, {"kv_len": 2})
    slot.compute_stream = object()
    with pytest.raises(RuntimeError, match="different streams"):
        decode.team_slot_launch(slot, 1)

    observed = []
    decode._input_preparation_plan = (
        SimpleNamespace(name="gather"), SimpleNamespace(name="positions"),
    )
    decode._spec.preparation_callbacks.update({
        "gather": lambda _slot, _batch: observed.append(
            ("gather", torch.is_inference_mode_enabled())),
        "positions": lambda _slot, _batch: observed.append(
            ("positions", torch.is_inference_mode_enabled())),
    })
    decode.prepare_team_inputs(slot, 1)
    assert observed == [("gather", True), ("positions", True)]


def test_collective_allocations_are_derived_from_all_descriptor_pairs():
    pairs = _collective_pairs(_descriptor())
    assert len(pairs) == 2
    assert {local["name"] for local, _ in pairs} == {
        "team_payload_local", "team_epoch_local"
    }
    assert {multicast["name"] for _, multicast in pairs} == {
        "team_payload_multicast", "team_epoch_multicast"
    }


@pytest.mark.parametrize("mutation", ("missing", "dtype", "transport", "duplicate"))
def test_collective_pair_parser_rejects_incomplete_abi(mutation):
    descriptor = _descriptor()
    buffers = descriptor["device_program"]["distribution_plan"][
        "routed_down_collectives"
    ][0]["buffers"]
    arguments = descriptor["device_program"]["argument_plan"]["arguments"]
    if mutation == "missing":
        buffers.pop(1)
    elif mutation == "dtype":
        arguments[0]["dtype"] = "fp32"
    elif mutation == "transport":
        arguments[0]["transport"] = "tensor"
    else:
        buffers[1]["role"] = buffers[0]["role"]
    with pytest.raises(RuntimeError):
        _collective_pairs(descriptor)


def test_rank_team_retains_each_owner_and_rejects_inconsistent_rank(monkeypatch):
    allocations = []

    class FakeAllocation:
        def __init__(self, devices, shape, *, dtype):
            assert tuple(devices) == tuple(range(8))
            self.shape = tuple(shape)
            self.dtype = dtype
            self.local = [object() for _ in devices]
            self.multicast = [object() for _ in devices]
            self.closed = False
            allocations.append(self)

        def close(self):
            self.closed = True

    multicast = ModuleType("kestrel_kernels.multicast")
    multicast.CudaMulticastAllocation = FakeAllocation
    monkeypatch.setitem(sys.modules, "kestrel_kernels.multicast", multicast)
    generated = ModuleType("kestrel_kernels.generated_decode")
    generated.BoundGeneratedDecodeRankTeam = _FakeBoundRankTeam
    monkeypatch.setitem(sys.modules, "kestrel_kernels.generated_decode", generated)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: _NullContext())
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    used_streams = []
    monkeypatch.setattr(
        torch.cuda, "stream",
        lambda stream: _RecordingContext(stream, used_streams),
    )
    members = generated_decode_team_members(range(8), topology="route_slot_ep")
    program = SimpleNamespace(descriptor=_descriptor())
    runtimes = []
    for member in members:
        decode = _FakeDecode(member, program)
        runtime = SimpleNamespace(
            device=torch.device("cuda", member.devices[member.rank]),
            max_batch_size=1,
            generated_decode=decode,
            decode_slots=(SimpleNamespace(
                slot_id=0, compute_stream=f"rank-{member.rank}-compute"
            ),),
        )
        runtime.decode_with_slot = lambda slot, batch, bound=decode: bound.launch(slot, batch)
        runtimes.append(runtime)
    team = GeneratedDecodeRankTeam(runtimes)
    assert [allocation.shape for allocation in allocations] == [(2, 8, 16), (2, 1)]
    assert [allocation.dtype for allocation in allocations] == [torch.bfloat16, torch.int32]
    assert all(len(runtime.generated_decode.inputs) == 4 for runtime in runtimes)
    assert team.queue_one() is runtimes[0].decode_slots[0]
    assert all(runtime.generated_decode.launches == 1 for runtime in runtimes)
    assert all(runtime.generated_decode.preparations == 1 for runtime in runtimes)
    assert set(used_streams) == {
        f"rank-{rank}-compute" for rank in range(8)
    }
    assert synchronized == []
    assert team._bound_team.launches == [(0, {"kv_len": 1})]
    team.wait()
    assert team._bound_team.waits == 1
    with pytest.raises(ValueError, match="slot0"):
        team.queue_one(slot_id=1)
    original_slots = runtimes[0].decode_slots
    runtimes[0].decode_slots = (SimpleNamespace(
        slot_id=0, compute_stream="replacement-stream",
    ),)
    with pytest.raises(RuntimeError, match="slot object changed"):
        team.queue_one()
    runtimes[0].decode_slots = original_slots
    runtimes[1].generated_decode.kv_len = 2
    with pytest.raises(RuntimeError, match="extents"):
        team.queue_one()
    assert team._bound_team.launches == [(0, {"kv_len": 1})]
    runtimes[1].generated_decode.kv_len = 1
    team.queue_one()
    assert team._bound_team.launches[-1] == (1, {"kv_len": 1})
    close_bound_team = team._bound_team.close
    team._bound_team.close = lambda: (_ for _ in ()).throw(
        RuntimeError("fatal packed rank team")
    )
    with pytest.raises(RuntimeError, match="fatal packed rank team"):
        team.close()
    assert all(not allocation.closed for allocation in allocations)
    assert all(runtime.generated_decode.inputs is not None for runtime in runtimes)
    team._bound_team.close = close_bound_team
    team.close()
    assert all(allocation.closed for allocation in allocations)
    assert all(runtime.generated_decode.inputs is None for runtime in runtimes)
    with pytest.raises(RuntimeError, match="closed"):
        team.queue_one()
    runtimes[7].generated_decode._spec.team_member = members[0]
    with pytest.raises(ValueError, match="inconsistent"):
        GeneratedDecodeRankTeam(runtimes)


def test_partial_bind_failure_unbinds_every_prior_rank_before_free(monkeypatch):
    allocations = []

    class FakeAllocation:
        def __init__(self, devices, _shape, *, dtype):
            self.local = [object() for _ in devices]
            self.multicast = [object() for _ in devices]
            self.closed = False
            allocations.append(self)

        def close(self):
            self.closed = True

    multicast = ModuleType("kestrel_kernels.multicast")
    multicast.CudaMulticastAllocation = FakeAllocation
    monkeypatch.setitem(sys.modules, "kestrel_kernels.multicast", multicast)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    members = generated_decode_team_members(range(8), topology="route_slot_ep")
    program = SimpleNamespace(descriptor=_descriptor())
    runtimes = []
    for member in members:
        decode = _FakeDecode(member, program)
        if member.rank == 2:
            decode.bind_team = lambda _inputs: (_ for _ in ()).throw(
                RuntimeError("synthetic rank bind failed")
            )
        runtimes.append(SimpleNamespace(
            device=torch.device("cuda", member.devices[member.rank]),
            max_batch_size=1,
            generated_decode=decode,
        ))
    with pytest.raises(RuntimeError, match="synthetic rank bind failed"):
        GeneratedDecodeRankTeam(runtimes)
    assert runtimes[0].generated_decode.inputs is None
    assert runtimes[1].generated_decode.inputs is None
    assert all(allocation.closed for allocation in allocations)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return None


class _RecordingContext(_NullContext):
    def __init__(self, stream, used_streams):
        self.stream = stream
        self.used_streams = used_streams

    def __enter__(self):
        self.used_streams.append(self.stream)


class _FakeDecode:
    def __init__(self, member, program):
        self._spec = SimpleNamespace(team_member=member)
        self.program = program
        self.inputs = None
        self.launches = 0
        self.preparations = 0
        self.kv_len = 1
        self._bound = None

    def team_program(self, _batch):
        return self.program

    def bind_team(self, inputs):
        self.inputs = inputs

    def unbind_team(self):
        self.inputs = None
        self._bound = None

    def team_slot_launch(self, slot, _batch):
        if self._bound is None:
            self._bound = _FakeBound(self, slot)
        return self._bound, {"kv_len": self.kv_len}

    def prepare_team_inputs(self, _slot, _batch):
        self.preparations += 1

    def launch(self, _slot, batch):
        assert batch == 1
        self.launches += 1


class _FakeBound:
    def __init__(self, decode, slot):
        self.decode = decode
        self.slot = slot
        self.stream = slot.compute_stream

    def invoke(self):
        self.decode.launch(self.slot, 1)


class _FakeBoundRankTeam:
    def __init__(self, devices, members, *, stage_step):
        self.devices = tuple(devices)
        self.members = tuple(members)
        self.stage_step = stage_step
        self.launches = []
        self.waits = 0
        self.closed = False

    def launch_step(self, step, **scalars):
        self.launches.append((step, scalars))
        for rank, member in enumerate(self.members):
            with torch.cuda.device(self.devices[rank]):
                with torch.cuda.stream(member.stream):
                    self.stage_step(rank, step)
                    member.invoke()

    def wait(self):
        self.waits += 1

    def close(self):
        self.closed = True
