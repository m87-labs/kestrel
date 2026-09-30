from argparse import Namespace
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from kestrel.models import get_spec
from kestrel.models.rfdetr.config import CONFIGS
from kestrel.models.rfdetr.preprocessing import preprocess_image
from kestrel.models.rfdetr.runtime import RFDetrRuntime, create_rfdetr_runtime, postprocess
from kestrel.models.rfdetr.weights import load_weights


def outputs():
    boxes = torch.full((300, 4), .5)
    logits = torch.full((300, 91), -100.)
    logits[0, 1] = 4
    logits[1, 90] = 3
    return boxes, logits


def test_sparse_coco_labels_and_xyxy():
    result = postprocess(*outputs(), max_objects=2)['objects']
    assert [x['class_id'] for x in result] == [1, 90]
    assert [x['label'] for x in result] == ['person', 'toothbrush']
    assert result[0]['x_min'] == .25 and result[0]['x_max'] == .75
    assert postprocess(*outputs(), threshold=1)['objects'] == []


def test_unused_logit_slots_not_reindexed_or_background_last():
    boxes, logits = outputs()
    logits[0, 0] = 8
    result = postprocess(boxes, logits)['objects']
    assert result[0]['class_id'] == 0 and result[0]['label'] == ''
    assert any(x['class_id'] == 90 for x in result)


def test_tensor_resize_matches_bilinear_cpu_without_aliasing():
    pixels = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7) / 105
    before = pixels.clone()
    actual = preprocess_image(pixels, 8)
    resized = torch.nn.functional.interpolate(pixels[None], (8, 8), mode='bilinear', align_corners=False, antialias=False)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    torch.testing.assert_close(actual, (resized - mean) / std)
    assert torch.equal(before, pixels)
    rgb = np.full((5, 7, 3), 127, np.uint8)
    assert torch.equal(preprocess_image(rgb, 8), preprocess_image(Image.fromarray(rgb), 8))


@pytest.mark.parametrize('image', [torch.ones(3, 2, 2) * float('nan'), torch.ones(3, 2, 2) * 256, np.zeros((2, 2), np.uint8)])
def test_invalid_pixels(image):
    with pytest.raises((ValueError, TypeError)):
        preprocess_image(image, 8)


def test_registry_all_current_variants_without_kv():
    assert len(CONFIGS) == 7
    for variant in CONFIGS:
        assert get_spec('rfdetr-' + variant).needs_kv_pool is False
    with pytest.raises(ValueError):
        get_spec('rfdetr-large-deprecated')


def test_lease_owned_until_cpu_result_created():
    class Executable:
        close_calls = 0
        active = False
        @contextmanager
        def borrow_outputs(self, pixels):
            assert pixels.dtype == torch.bfloat16
            self.active = True
            boxes, logits = outputs()
            yield boxes, logits
            boxes.fill_(float('nan'))
            logits.fill_(float('nan'))
            self.active = False
        def close(self):
            self.close_calls += 1
    executable = Executable()
    runtime = RFDetrRuntime(name='rfdetr-nano', config=CONFIGS['nano'], executable=executable, device=torch.device('cpu'))
    result, = runtime.forward('detect', ({'image': Image.new('RGB', (4, 4))},))
    assert result['objects'][0]['label'] == 'person'
    assert not executable.active
    runtime.shutdown()
    runtime.shutdown()
    assert executable.close_calls == 1
    with pytest.raises(RuntimeError):
        runtime.forward('detect', ({'image': Image.new('RGB', (4, 4))},))


def test_unsupported_device_rejected_before_weights(monkeypatch):
    import kestrel.models.rfdetr.runtime as module
    monkeypatch.setattr(module, 'load_weights', lambda *a: pytest.fail('weights loaded on unsupported device'))
    cfg = SimpleNamespace(resolved_device=lambda: torch.device('cpu'), resolved_dtype=lambda: torch.bfloat16)
    with pytest.raises(ValueError, match='Hopper'):
        create_rfdetr_runtime(cfg)


def test_safe_checkpoint_tensor_extraction(tmp_path):
    path = tmp_path / 'weights.pth'
    state = {'a': torch.ones(2)}
    torch.save({'model': state, 'args': Namespace(resolution=384)}, path)
    assert torch.equal(load_weights('nano', path)['a'], state['a'])
    torch.save({'model': {'a': 1}}, path)
    with pytest.raises(ValueError):
        load_weights('nano', path)


def test_detect_handle_routes_single_pass_without_prompt_adaptation():
    import asyncio
    from kestrel.engine import InferenceEngine
    from kestrel.runtime import ExecutionShape
    engine = object.__new__(InferenceEngine)
    engine._default_model = 'rfdetr-nano'
    engine._model_ids = ['rfdetr-nano']
    engine._runtimes = {'rfdetr-nano': SimpleNamespace(
        model_name='rfdetr-nano', execution_shape=ExecutionShape.SINGLE_PASS,
        tasks=lambda: ('detect',))}
    engine._initialized = True
    engine._scheduler_error = None
    captured = {}
    async def run(model, task, inputs):
        captured.update(model=model, task=task, inputs=inputs)
        return {'objects': []}
    engine.run = run
    image = Image.new('RGB', (4, 4))
    result = asyncio.run(engine.model('rfdetr-nano').detect(image=image, threshold=.7))
    assert result == {'objects': []}
    assert captured == {'model': 'rfdetr-nano', 'task': 'detect',
                        'inputs': {'image': image, 'threshold': .7}}


def test_padded_output_carriers_cropped_only_after_readback():
    class Executable:
        @contextmanager
        def borrow_outputs(self, pixels):
            boxes = torch.full((384, 128), float('nan'))
            logits = torch.full((384, 128), float('nan'))
            b, l = outputs()
            boxes[:300, :4], logits[:300, :91] = b, l
            yield boxes, logits
    runtime = RFDetrRuntime(name='rfdetr-nano', config=CONFIGS['nano'],
                            executable=Executable(), device=torch.device('cpu'))
    result, = runtime.forward('detect', ({'image': Image.new('RGB', (4, 4))},))
    assert result['objects'][0]['class_id'] == 1


@pytest.mark.parametrize('options', [{'object': 'cat'}, {'threshold': float('nan')},
                                     {'threshold': True}, {'max_objects': 0}, {'max_objects': 301}])
def test_invalid_request_fails_before_launch(options):
    runtime = RFDetrRuntime(name='rfdetr-nano', config=CONFIGS['nano'],
                            executable=None, device=torch.device('cpu'))
    with pytest.raises(ValueError):
        runtime.forward('detect', ({'image': Image.new('RGB', (4, 4)), **options},))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 1e39, -1e39])
def test_cpu_pixels_must_remain_finite_after_conversion(value):
    runtime = RFDetrRuntime(name='rfdetr-nano', config=CONFIGS['nano'],
                           executable=None, device=torch.device('cpu'))
    pixels = torch.zeros((1, 3, 384, 384), dtype=torch.float64)
    pixels[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match='finite after BF16 conversion'):
        runtime.forward('detect', ({'pixel_values': pixels},))
