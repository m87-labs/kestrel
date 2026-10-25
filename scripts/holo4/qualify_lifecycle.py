"""Exercise installed speculative serving through its public streaming API.

These are serving and answer-integrity checks, not an agent task-success eval.
Run from outside the source checkout with candidate wheels installed.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path


async def run(args):
    import torch
    import kestrel
    import kestrel_kernels
    from transformers import AutoTokenizer
    from kestrel.config import RuntimeConfig
    from kestrel.engine import InferenceEngine
    from kestrel.models.qwen35.skills import Qwen35ChatSkill
    from kestrel.runtime.tokens import TextToken
    from kestrel.skills import SkillRegistry

    class RawChat(Qwen35ChatSkill):
        def build_prompt_tokens(self, runtime, request_context):
            return [TextToken(x) for x in runtime.tokenizer.encode(
                request_context.messages[0].text).ids]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kv_cache_pages = args.kv_cache_pages or 32768 * args.batch_capacity
    engine = await InferenceEngine.create(RuntimeConfig(
        model='Qwen/Qwen3.6-35B-A3B', model_path=args.model, tokenizer_path=args.model,
        draft_model_path=args.draft, device='cuda', dtype=torch.bfloat16,
        max_batch_size=args.batch_capacity, page_size=1,
        kv_cache_pages=kv_cache_pages,
        enable_prefix_cache=False, decode_path=args.decode_path, verification_path=args.verification_path), skills=SkillRegistry([RawChat()]))
    report = dict(complete=False, task_success_evaluated=False, checks=[],
                  runtime=kestrel.__file__, kernels=kestrel_kernels.__file__,
                  batch_capacity=args.batch_capacity, decode_path=args.decode_path, verification_path=args.verification_path,
                  kv_cache_pages=kv_cache_pages)

    async def stream(prompt, limit=64, temperature=0):
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return await engine.chat(messages=[{'role': 'user', 'content': rendered}],
            reasoning=False, stream=True,
            settings={'temperature': temperature, 'max_tokens': limit, 'top_p': 1})

    async def finish(prompt, limit=64):
        value = await stream(prompt, limit)
        async for _ in value:
            pass
        result = await value.result()
        assert len(result.tokens) <= limit
        assert result.metrics.output_tokens == len(result.tokens)
        return tokenizer.decode([t.token_id for t in result.tokens], skip_special_tokens=True).strip()

    def record(name, **details):
        report['checks'].append(dict(name=name, passed=True, **details))
        args.output.write_text(json.dumps(report, indent=2))
        print(name, 'passed', flush=True)

    # Observe real serving commits, outside timed qualification. Require a direct
    # staged launch at the advertised cohort size, not merely a bound kernel.
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    observed_dispatch = {}
    original_commit = DFlashDraftGraphSession._commit_context
    def checked_commit(session, lengths):
        result = original_commit(session, lengths)
        dispatch = session.dispatch_report
        copy = dispatch["context_copy"]
        assert copy["implementation"] == "packed", dispatch
        assert copy["fallback_reason"] is None, dispatch
        assert len(copy["launch_plan"]["ranges"]) == 1, dispatch
        if dispatch["staging_implementation"] is not None:
            if dispatch["owners"] > 1:
                assert dispatch["staging_implementation"] == "packed", dispatch
                assert len(dispatch["staging_launch_plan"]["ranges"]) == 1, dispatch
            observed_dispatch[dispatch["owners"]] = dispatch
        return result
    DFlashDraftGraphSession._commit_context = checked_commit

    try:
        assert engine.runtime.spec.decoder._generated_verification is not None
        canaries = [('What is 2 + 2? Reply with only the numeral.', '4'),
                    ('Copy exactly this string and nothing else: RELEASE_CANARY_73', 'RELEASE_CANARY_73')]
        for prompt, expected in canaries:
            actual = await finish(prompt)
            assert actual == expected, (prompt, actual, expected)
        record('answer_canaries')
        for limit in (1, 2, 3, 5, 7, 8, 15, 16, 17, 31):
            await finish('List the integers from 1 to 1000 in ascending order.', limit)
        record('token_limits_cross_verification_boundaries')
        for index in range(8):
            active = [await stream('List the integers from 1 to 1000 in ascending order.', 512)
                      for _ in range(args.batch_capacity)]
            await asyncio.gather(*(value.__anext__() for value in active))
            queued = await stream(canaries[1][0])
            await asyncio.gather(queued.aclose(), *(value.aclose() for value in active))
            assert (await queued.result()).finish_reason == 'cancelled'
            assert all(result.finish_reason == 'cancelled'
                       for result in await asyncio.gather(*(value.result() for value in active)))
            assert await finish(canaries[0][0]) == '4'
        record('active_and_queued_cancellation', cycles=8)
        rejected = await stream('Say hello.', temperature=.5)
        try:
            async for _ in rejected:
                pass
            await rejected.result()
        except (ValueError, RuntimeError) as exc:
            assert "Qwen DFlash supports greedy text" in str(exc), str(exc)
            try:
                await rejected.result()
            except (ValueError, RuntimeError) as result_error:
                assert "Qwen DFlash supports greedy text" in str(result_error)
        else:
            raise AssertionError('unsupported sampling must be rejected')
        assert await finish(canaries[1][0]) == canaries[1][1]
        record('unsupported_request_recovery')
        for _ in range(3):
            await asyncio.gather(*(finish(canaries[i % 2][0]) for i in range(8)))
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
        for repeat in range(args.rounds):
            results = await asyncio.gather(*(finish(canaries[i % 2][0]) for i in range(8)))
            assert results == [canaries[i % 2][1] for i in range(8)], results
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated()
        assert memory_after <= memory_before + 8*1024*1024, (memory_before, memory_after)
        assert not engine.runtime.spec.decoder._sessions
        assert engine.runtime.spec.decoder.free_slots == args.batch_capacity
        record('request_isolation_and_memory_stability', requests=args.rounds*8,
               memory_before=memory_before, memory_after=memory_after)
        compiler = [name for name in sys.modules if name.startswith(
            ('mkl.compiler', 'mkl.megakernel', 'scripts.precompile'))]
        assert not compiler, compiler
        # Attention and pointwise Triton kernels may JIT; this gate covers
        # serving without the private megakernel compiler/build toolchain.
        record('no_megakernel_compiler_dependency')
        # Short answer canaries may finish before a full batch is admitted.
        # Exercise sustained concurrent decode explicitly rather than weakening
        # the dispatch coverage requirement when those canaries only observe C1.
        for limit in (512, 2048):
            if args.batch_capacity in observed_dispatch:
                break
            await asyncio.gather(*(finish(
                'List the integers from 1 to 10000 in ascending order.', limit)
                for _ in range(args.batch_capacity)))
        assert args.batch_capacity in observed_dispatch, observed_dispatch
        record('packed_draft_dispatch', observed=observed_dispatch)
        report['complete'] = True
        args.output.write_text(json.dumps(report, indent=2))
    finally:
        DFlashDraftGraphSession._commit_context = original_commit
        await engine.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--kv-cache-pages', type=int)
    parser.add_argument('--batch-capacity', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--decode-path', choices=('auto', 'generated'), default='auto')
    parser.add_argument('--verification-path', choices=('auto', 'native', 'generated'), default='auto')
    asyncio.run(asyncio.wait_for(run(parser.parse_args()), timeout=1800))
