"""Qualify one model-declared retained-prefix requirement with installed CUDA wheels.

This is a release gate for the prefix primitive, not whole-model coverage or a
performance test. It loads configuration only, never checkpoint weight shards.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--model-config', type=Path, required=True)
p.add_argument('--draft-config', type=Path, required=True)
p.add_argument('--capacity', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
# The executable gate must use installed packages, never a checkout overlay.
os.environ['KESTREL_CUTE_JIT'] = '0'
os.environ['KESTREL_REQUIRE_NATIVE'] = '1'
sys.modules['cutlass'] = None
sys.modules['triton'] = None
import importlib.metadata
import torch
import kestrel
import kestrel_kernels
from dataclasses import asdict
from kestrel.models.qwen35.qwen_config import Qwen3_5Config
from kestrel.models.qwen35.dflash import DFlashConfig
from kestrel_kernels import get_runtime
from kestrel_kernels.cubin_runtime import capture_packed_artifact_receipts
package_provenance = {}
for distribution, module in (('kestrel', kestrel), ('kestrel-kernels', kestrel_kernels)):
    dist = importlib.metadata.distribution(distribution)
    installed = Path(dist.locate_file(module.__name__)).resolve()
    actual = Path(module.__file__).resolve().parent
    if actual != installed:
        raise RuntimeError(f'{distribution} must be imported from its installed wheel: {actual} != {installed}')
    record = next((entry for entry in dist.files or () if str(entry).endswith('.dist-info/RECORD')), None)
    if record is None:
        raise RuntimeError(f'{distribution} has no installed wheel RECORD')
    package_provenance[distribution] = dict(version=dist.version, package_path=str(actual), module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), record_sha256=hashlib.sha256(Path(dist.locate_file(record)).read_bytes()).hexdigest())
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for native retained-prefix coverage qualification')
model = Qwen3_5Config.from_dict(json.loads(a.model_config.read_text())).text_config
draft = DFlashConfig.from_dict(json.loads(a.draft_config.read_text()))
demand = model.retained_prefix_demand(capacity=a.capacity, verification_width=draft.block_size)
width = demand.dimension('verification_width')
nk = demand.dimension('num_k_heads')
nv = demand.dimension('num_v_heads')
dk = demand.dimension('key_head_dim')
dv = demand.dimension('value_head_dim')
bounds = dict(demand.dynamic_bounds)
if dk != dv:
    raise RuntimeError(f'retained-prefix gate requires equal key/value head dimensions, got {dk}/{dv}')
r = get_runtime('cuda').gated_delta
checks = []
receipts = []
for count in range(bounds['sequences'].lower, bounds['sequences'].upper + 1):
    torch.manual_seed(190 + count)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with capture_packed_artifact_receipts() as artifacts, torch.cuda.stream(stream):
        mixed = torch.randn((1, count*width, 2*nk*dk+nv*dv), device='cuda', dtype=torch.bfloat16)*.1
        gate = torch.randn((1, count*width, nv), device='cuda', dtype=torch.bfloat16)
        beta = torch.randn_like(gate)
        log_decay = torch.zeros(nv, device='cuda', dtype=torch.float32)
        bias = torch.zeros_like(log_decay)
        initial = torch.randn((count,nv,dv,dk),device='cuda',dtype=torch.bfloat16)*.01
        saved = initial.clone()
        lengths = torch.full((count,),width,device='cuda',dtype=torch.int32)
        state = torch.empty_like(initial)
        cu, topology = r.bind_packed_prefill_topology(sequence_lengths=(width,)*count,device=mixed.device)
        workspace = r.allocate_packed_gated_delta_prefill_workspace(mixed, gate, head_dim=dk)
        context = r.allocate_packed_gated_delta_prefix_context(workspace,initial,cu,sequence_lengths=(width,)*count,topology_token=topology)
        if not context.supports_graph_capture:
            raise RuntimeError(f'missing native prefix runtime route: {nk=}, {nv=}, {dk=}, {dv=}, {width=}, {count=}')
        stages = {}
        with capture_packed_artifact_receipts() as prepare_artifacts:
            r.packed_prefill_prepare(mixed,gate,beta,log_decay,bias,query=workspace.query,key=workspace.key,value=workspace.value,g=workspace.log_g,beta=workspace.beta_fp32)
        stages['prepare'] = prepare_artifacts.receipts
        def execute():
            with capture_packed_artifact_receipts() as verify_artifacts:
                out,_ = r.packed_gated_delta_rule_prefill(mixed,gate,beta,log_decay,bias,cu,workspace=workspace,initial_state=initial,output_final_state=False,sequence_lengths=(width,)*count,topology_token=topology,prefix_context=context)
            stages['verify'] = verify_artifacts.receipts
            with capture_packed_artifact_receipts() as finalize_artifacts:
                r.finalize_packed_gated_delta_prefix([context],lengths,out_states=[state])
            stages['finalize'] = finalize_artifacts.receipts
            return out
        out = execute()
        expected_out = out.clone()
        expected_states = {}
        for accepted in range(bounds['accepted_length'].lower, bounds['accepted_length'].upper + 1):
            lengths.fill_(accepted)
            execute()
            expected_states[accepted] = state.clone()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            actual = execute()
        for accepted, expected_state in expected_states.items():
            lengths.fill_(accepted)
            graph.replay()
            torch.testing.assert_close(actual,expected_out,rtol=0,atol=0)
            torch.testing.assert_close(state,expected_state,rtol=0,atol=0)
        torch.testing.assert_close(initial,saved,rtol=0,atol=0)
        stream.synchronize()
    if not artifacts.receipts:
        raise RuntimeError(f'no native artifact receipts for cohort {count}')
    for stage, evidence in stages.items():
        if not evidence:
            raise RuntimeError(f'missing native {stage} artifact evidence for cohort {count}')
    receipts.extend(artifacts.receipts)
    checks.append(dict(cohort=count,width=width,graph_replay=True,initial_state_preserved=True,stage_artifacts=stages))
    del graph, context, workspace, initial, saved, expected_states, actual, out, state
    torch.cuda.empty_cache()
result = dict(complete=True,compiler_blocked=True,graph_consistency=True,numerical_correctness=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),installed_packages=package_provenance,model_config_sha256=hashlib.sha256(a.model_config.read_bytes()).hexdigest(),draft_config_sha256=hashlib.sha256(a.draft_config.read_bytes()).hexdigest(),requirement=asdict(demand),scope="retained_prefix_only",whole_model_coverage=False,checks=checks,artifacts=receipts)
a.output.write_text(json.dumps(result,indent=2))
print('Declared native prefix routes passed',len(checks),'cohorts',flush=True)
