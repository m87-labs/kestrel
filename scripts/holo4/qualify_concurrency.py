"""Measure closed-loop client concurrency separately from GPU batch capacity.

Run in the selected installed runtime environment. Optional bootstrap is for
explicitly labelled development artifacts only; release qualification must omit it.
Clock probes surround complete windows, never individual concurrent requests.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
import importlib.metadata
import time
from pathlib import Path


def valid_clock(probe):
    return (probe['n_sm'] == probe['expected_n_sm']
            and abs(probe['state_mhz'] - 1605) <= .5
            and probe['max_deviation_mhz'] < 7.5 and probe['spread_mhz'] < 12)


def cohort_decode_window(rows):
    """Token delivery while the complete cohort is decoding, with no admissions.

    Exclude the boundary delivery that establishes the final request's TTFT.
    Stop at the earliest request completion; never add overlapping durations.
    This is a client-observed aggregate rate, not kernel-only execution time.
    """
    if not rows or any(not row['decode_events'] for row in rows):
        raise ValueError('every cohort request must deliver tokens')
    start = max(row['decode_events'][0][0] for row in rows)
    end = min(row['decode_events'][-1][0] for row in rows)
    if end <= start:
        raise ValueError('cohort has no shared decode interval')
    tokens = sum(count for row in rows for stamp, count in row['decode_events']
                 if start < stamp <= end)
    return dict(start=start, end=end, elapsed_s=end-start, output_tokens=tokens,
                active_requests=len(rows), output_tokens_s=tokens/(end-start))


async def run(args):
    if args.cohort_decode and max(args.concurrencies) > args.batch_capacity:
        raise ValueError('decode cohorts must fit GPU batch capacity')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    manifest = json.loads(args.manifest.read_text())
    items = [dict(row, prompt=tokenizer.apply_chat_template(
        row['messages'], tokenize=False, add_generation_prompt=True,
        enable_thinking=True)) for row in manifest['requests'] if not row['image']]
    if args.bootstrap:
        spec = importlib.util.spec_from_file_location('qualification_bootstrap', args.bootstrap)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if args.engine == 'kestrel':
        import torch
        from kestrel.config import RuntimeConfig
        from kestrel.engine import InferenceEngine
        from kestrel.models.qwen35.skills import Qwen35ChatSkill
        from kestrel.runtime.tokens import TextToken
        from kestrel.skills import SkillRegistry
        from mkl.bench.clocks import kernel_observed_sm_mhz

        class RawChat(Qwen35ChatSkill):
            def build_prompt_tokens(self, runtime, request_context):
                return [TextToken(int(x)) for x in runtime.tokenizer.encode(
                    request_context.messages[0].text).ids]

        engine = await InferenceEngine.create(RuntimeConfig(
            model='Qwen/Qwen3.6-35B-A3B', model_path=args.model,
            tokenizer_path=args.model, draft_model_path=args.draft,
            device='cuda', dtype=torch.bfloat16, max_batch_size=args.batch_capacity,
            page_size=1, kv_cache_pages=args.kv_cache_pages, enable_prefix_cache=False,
            decode_path=args.decode_path), skills=SkillRegistry([RawChat()]))

        async def probe():
            return kernel_observed_sm_mhz(ramp_ms=200)

        async def align():
            from benchmark_clock_control_settled import align_clock
            return align_clock(1605, 1605, device_index=1)

        async def generate(item, request_id):
            started = time.perf_counter()
            first = None
            events = []
            stream = await engine.chat(messages=[{'role': 'user', 'content': item['prompt']}],
                reasoning=False, stream=True,
                settings={'temperature': 0, 'max_tokens': args.tokens, 'top_p': 1})
            async for update in stream:
                if update.token is not None:
                    stamp = time.perf_counter()
                    if args.cohort_decode:
                        events.append((stamp, 1))
                    if first is None:
                        first = stamp - started
            result = await stream.result()
            elapsed = time.perf_counter()-started
            return dict(decode_events=events, elapsed_ms=1000*elapsed,
                        ttft_ms=None if first is None else first*1000,
                        decode_wall_ms=None if first is None else 1000*(elapsed-first),
                        first_output_tokens=1 if first is not None else 0,
                        model_decode_ms=result.metrics.decode_time_ms,
                        output_ids=[t.token_id for t in result.tokens],
                        finish_reason=result.finish_reason)

        async def shutdown():
            await engine.shutdown()
    else:
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        speculative = {} if args.draft is None else {'speculative_config': {
            'method': 'dflash', 'model': str(args.draft), 'num_speculative_tokens': args.num_speculative_tokens}}
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=str(args.model), dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=32768, max_num_seqs=args.batch_capacity,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens, enable_prefix_caching=False,
            mm_processor_cache_gb=0, limit_mm_per_prompt={'image': 1},
            mm_processor_kwargs={'min_pixels': 65536, 'max_pixels': 16777216},
            generation_config='vllm', seed=20261008,
            worker_extension_cls='vllm_benchmark_worker_settled.ClockProbeExtension',
            **speculative))

        async def probe():
            readings = await engine.collective_rpc('benchmark_clock')
            assert len(readings) == 1
            return readings[0]['probe']

        async def align():
            return await engine.collective_rpc('benchmark_align_clock', args=(1605, 1605))

        async def generate(item, request_id):
            started = time.perf_counter()
            first = None
            events = []
            final = None
            first_output_tokens = 0
            delivered = 0
            async for result in engine.generate({'prompt': item['prompt']},
                    SamplingParams(temperature=0, max_tokens=args.tokens), str(request_id)):
                count = len(result.outputs[0].token_ids)
                if args.cohort_decode and count > delivered:
                    events.append((time.perf_counter(), count-delivered))
                delivered = count
                if first is None and result.outputs[0].token_ids:
                    first = time.perf_counter()-started
                    first_output_tokens = len(result.outputs[0].token_ids)
                final = result
            elapsed = time.perf_counter()-started
            return dict(decode_events=events, elapsed_ms=1000*elapsed,
                        ttft_ms=None if first is None else first*1000,
                        decode_wall_ms=None if first is None else 1000*(elapsed-first),
                        first_output_tokens=first_output_tokens,
                        output_ids=list(final.outputs[0].token_ids),
                        finish_reason=final.outputs[0].finish_reason)

        async def shutdown():
            engine.shutdown()

    report = dict(engine=args.engine, batch_capacity=args.batch_capacity,
                  cohort_decode=args.cohort_decode,
                  gpu_memory_utilization=args.gpu_memory_utilization if args.engine == "vllm" else None,
                  max_num_batched_tokens=args.max_num_batched_tokens if args.engine == "vllm" else None,
                  num_speculative_tokens=args.num_speculative_tokens if args.engine == "vllm" and args.draft else None,
                  model=str(args.model), draft=str(args.draft), tokens=args.tokens,
                  bootstrap=str(args.bootstrap), decode_path=args.decode_path,
                  kv_cache_pages=args.kv_cache_pages, complete=False, windows=[], attempts=[])
    versions = {}
    for package in ('torch', 'vllm', 'kestrel', 'kestrel-kernels', 'transformers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report['environment'] = dict(python=sys.version, packages=versions,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    counter = 0

    async def window(concurrency):
        nonlocal counter
        semaphore = asyncio.Semaphore(concurrency)

        async def one(item):
            nonlocal counter
            async with semaphore:
                counter += 1
                row = await generate(item, counter)
                return dict(row, id=item['id'], prompt_sha256=hashlib.sha256(
                    item['prompt'].encode()).hexdigest())

        started = time.perf_counter()
        decode_windows = []
        if args.cohort_decode:
            rows = []
            for index in range(0, len(items), concurrency):
                cohort = await asyncio.gather(*(one(item) for item in items[index:index+concurrency]))
                try:
                    interval = dict(cohort_decode_window(cohort), available=True)
                except ValueError as exc:
                    if str(exc) != 'cohort has no shared decode interval':
                        raise
                    interval = dict(available=False, reason=str(exc), active_requests=len(cohort))
                interval['request_ids'] = [row['id'] for row in cohort]
                decode_windows.append(interval)
                rows.extend(cohort)
        else:
            rows = await asyncio.gather(*(one(item) for item in items))
        elapsed = time.perf_counter()-started
        return dict(concurrency=concurrency, elapsed_s=elapsed, rows=rows,
                    decode_windows=decode_windows,
                    aggregate_decode_tokens_s=(sum(w["output_tokens"] for w in decode_windows) /
                        sum(w["elapsed_s"] for w in decode_windows) if decode_windows and all(w["available"] for w in decode_windows) else None),
                    output_tokens_s=sum(len(row['output_ids']) for row in rows)/elapsed,
                    requests_s=len(rows)/elapsed)

    try:
        for concurrency in args.concurrencies:
            await window(concurrency)
            report.setdefault('clock_control', []).append(await align())
            for repeat in range(args.repeats):
                for attempt in range(4):
                    before = await probe()
                    result = await window(concurrency)
                    after = await probe()
                    result.update(repeat=repeat, attempt=attempt,
                                  clock_before=before, clock_after=after)
                    result['clock_qualified'] = valid_clock(before) and valid_clock(after)
                    report['attempts'].append(result)
                    if result['clock_qualified']:
                        report['windows'].append(result)
                    args.output.write_text(json.dumps(report, indent=2))
                    print(json.dumps({k: v for k, v in result.items()
                                      if k not in ('rows', 'clock_before', 'clock_after', 'decode_windows')}), flush=True)
                    if result['clock_qualified']:
                        break
                    report['clock_control'].append(await align())
                else:
                    raise RuntimeError('four clock-invalid windows')
        report['complete'] = True
        args.output.write_text(json.dumps(report, indent=2))
    finally:
        await shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('kestrel', 'vllm'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--draft', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-capacity', type=int, required=True)
    parser.add_argument('--concurrencies', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num-speculative-tokens', type=int, default=15)
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.92)
    parser.add_argument('--cohort-decode', action='store_true')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--bootstrap', type=Path)
    parser.add_argument('--decode-path', choices=('auto', 'generated'), default='auto')
    parser.add_argument('--kv-cache-pages', type=int, default=32768)
    asyncio.run(run(parser.parse_args()))
