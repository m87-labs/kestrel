"""Exercise the public image API using installed, compiler-free GPU artifacts.

Run on Hopper with a checkpoint and an image, for example:
    python scripts/smoke_single_pass_vision.py --model rfdetr-nano \
        --checkpoint /models/rf-detr-nano.pth --image /images/example.jpg

This is an API/ownership/launch smoke, not a replacement for model quality gates.
"""
import argparse
import asyncio
import importlib.abc
import json
import math
from pathlib import Path
import sys


class NoCompilerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mkl', 'rfdetr', 'rfdetr_plus', 'kestrel_rfdetr', 'kestrel_dinov2'}:
            raise RuntimeError(f'Public serving imported a build/training dependency: {fullname}')
        return None


async def smoke(args):
    import torch
    from PIL import Image, ImageOps
    from kestrel.config import RuntimeConfig
    from kestrel.engine import InferenceEngine

    cfg = RuntimeConfig(model=args.model, model_path=args.checkpoint,
                        device='cuda:0', dtype=torch.bfloat16)
    engine = await InferenceEngine.create(cfg)
    image = Image.open(args.image).convert('RGB')
    model = engine.model(args.model)
    detection = args.model.startswith('rfdetr-')

    async def infer(value):
        result = await (model.detect(image=value) if detection else model.embed(image=value))
        if detection:
            output = result.output
            for obj in output['objects']:
                for field in ('x_min', 'y_min', 'x_max', 'y_max', 'score'):
                    assert math.isfinite(obj[field]) and 0 <= obj[field] <= 1, (field, obj)
            return output
        hidden = result.output['last_hidden_state'].cpu()
        pooled = result.output['pooler_output'].cpu()
        assert hidden.shape == (1, 257, 384) and pooled.shape == (1, 384)
        assert torch.isfinite(hidden).all()
        torch.testing.assert_close(pooled, hidden[:, 0])
        return hidden

    try:
        first = await infer(image)
        await infer(ImageOps.invert(image))
        # Kernel census is for the public forward, including transfers and
        # postprocessing, after startup has created descriptors and buffers.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            repeated = await infer(image)
            torch.cuda.synchronize()
        kernels = [e.name for e in prof.events()
                   if e.device_type == torch.autograd.DeviceType.CUDA
                   and not e.name.startswith(('Memcpy', 'Memset'))]
        assert len(kernels) == 1, kernels
        if detection:
            # Repeated-input stability is distinct from independent correctness.
            a, b = first['objects'], repeated['objects']
            assert len(a) == len(b)
            key = lambda o: (o['class_id'], o['x_min'], o['y_min'], o['x_max'], o['y_max'])
            for x, y in zip(sorted(a, key=key), sorted(b, key=key)):
                assert x['class_id'] == y['class_id']
                torch.testing.assert_close(torch.tensor([x[k] for k in ('x_min','y_min','x_max','y_max','score')]),
                                           torch.tensor([y[k] for k in ('x_min','y_min','x_max','y_max','score')]),
                                           rtol=.01, atol=.005)
        else:
            torch.testing.assert_close(first, repeated, rtol=.01, atol=.005)
        report = {'model': args.model, 'kernels': kernels, 'repeat_stable': True,
                  'compiler_imports': sorted(n for n in sys.modules if n == 'mkl' or n.startswith('mkl.')),
                  'scope': 'Public API smoke and repeated-input stability; independent numerical qualification is separate.'}
        assert not report['compiler_imports']
        if detection:
            report['output'] = repeated
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        await engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.meta_path.insert(0, NoCompilerImports())
    asyncio.run(smoke(args))


if __name__ == '__main__':
    main()
