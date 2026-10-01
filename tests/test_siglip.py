from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from kestrel.models import get_spec, known_models
from kestrel.models.siglip.preprocessing import preprocess_image
from kestrel.models.siglip.runtime import SiglipRuntime, create_siglip_runtime
from kestrel.models.siglip.weights import load_siglip_weights
from kestrel.models.moondream.weights import md3_vision_parameter_sources


def cfg(device='cpu', path=None):
    return SimpleNamespace(model='siglip-so400m-378', model_path=path,
                           resolved_device=lambda: torch.device(device),
                           resolved_dtype=lambda: torch.bfloat16)


class Executable:
    def __init__(self):
        self.calls = []
        self.closed = 0
    def encode_crops(self, pixels):
        self.calls.append(pixels)
        return torch.full((len(pixels), 729, 1152), len(self.calls), dtype=torch.bfloat16)
    def close(self):
        self.closed += 1


def test_registration_and_owned_batch_outputs():
    assert 'siglip-so400m-378' in known_models()
    spec = get_spec('siglip-so400m-378')
    assert spec.runtime is create_siglip_runtime and not spec.needs_kv_pool
    executable = Executable()
    runtime = SiglipRuntime(cfg(), executable=executable)
    outputs = []
    for n in (1, 2, 13):
        output = runtime.forward('embed', ({'pixel_values': torch.zeros(n, 3, 378, 378, dtype=torch.uint8)},))[0]
        assert set(output) == {'last_hidden_state'}
        outputs.append(output['last_hidden_state'])
        assert tuple(outputs[-1].shape) == (n, 729, 1152)
    assert torch.all(outputs[0] == 1)
    runtime.shutdown()
    runtime.shutdown()
    assert executable.closed == 1
    with pytest.raises(RuntimeError, match='shut down'):
        runtime.forward('embed', ({},))


@pytest.mark.parametrize('inputs,message', [
    ({}, 'exactly one'),
    ({'image': None, 'pixel_values': torch.empty(0)}, 'exactly one'),
    ({'pixel_values': torch.zeros(1, 3, 378, 378)}, 'uint8'),
    ({'pixel_values': torch.zeros(14, 3, 378, 378, dtype=torch.uint8)}, 'shape'),
    ({'pixel_values': torch.zeros(1, 3, 384, 384, dtype=torch.uint8)}, 'shape'),
    ({'pixel_values': torch.empty(1, 3, 378, 378, device='meta', dtype=torch.uint8)}, 'model device'),
    ({'pixel_values': None}, 'torch.Tensor'),
    ({'image': object()}, 'PIL image'),
    ({'image': object(), 'normalize': True}, 'accepts image'),
])
def test_bad_inputs_do_not_launch(inputs, message):
    executable = Executable()
    runtime = SiglipRuntime(cfg(), executable=executable)
    with pytest.raises((ValueError, TypeError), match=message):
        runtime.forward('embed', (inputs,))
    assert executable.calls == []


def test_single_image_preserves_raw_pixels_until_kernel():
    image = Image.fromarray(np.full((6, 9, 3), 173, dtype=np.uint8))
    raw = preprocess_image(image)
    assert raw.dtype is torch.uint8 and raw.shape == (1, 3, 378, 378)
    assert raw.is_contiguous() and torch.all(raw == 173)
    runtime = SiglipRuntime(cfg(), executable=Executable())
    output = runtime.forward('embed', ({'image': image},))[0]['last_hidden_state']
    assert output.shape == (1, 729, 1152)
    torch.testing.assert_close(runtime.preprocess_image_async(image).result(), raw)
    with pytest.raises(TypeError):
        runtime.preprocess_image_async(object()).result()


def test_factory_rejects_unsupported_target_or_missing_checkpoint_before_load(monkeypatch):
    monkeypatch.setattr('kestrel.models.siglip.runtime.load_siglip_weights',
                        lambda path: pytest.fail('must not load'))
    with pytest.raises(ValueError, match='Hopper'):
        create_siglip_runtime(cfg())
    monkeypatch.setattr('kestrel.models.siglip.runtime.get_device_capability', lambda device: (9, 0))
    with pytest.raises(ValueError, match='model_path'):
        create_siglip_runtime(cfg('cuda:0'))


def test_factory_passes_state_directly_without_native_model(monkeypatch):
    calls, state = [], {'patch_emb.weight': torch.ones(2)}
    monkeypatch.setattr('kestrel.models.siglip.runtime.get_device_sm_count', lambda device: 132)
    monkeypatch.setattr('kestrel.models.siglip.runtime.load_siglip_weights', lambda path: state)
    monkeypatch.setattr('kestrel.models.siglip.runtime.get_device_capability', lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    def create(**kwargs):
        calls.append(kwargs)
        return Executable()
    monkeypatch.setitem(sys.modules, 'kestrel_kernels.megakernel.siglip',
                        SimpleNamespace(SiglipTokenEncoder=create))
    runtime = create_siglip_runtime(cfg('cuda:0', 'model_fp8.pt'))
    assert calls[0]['state_dict'] is state
    assert calls[0]['config'].crop_size == 378
    assert calls[0]['device'] == torch.device('cuda:0')
    assert runtime.tasks() == ('embed',)


def test_shared_checkpoint_mapping_excludes_projection_for_encoder():
    encoder = md3_vision_parameter_sources(27, include_projection=False)
    full = md3_vision_parameter_sources(27)
    assert len(encoder) == 5 + 27 * 12
    assert set(full) - set(encoder) == {
        'proj_mlp.fc1.weight', 'proj_mlp.fc1.bias', 'proj_mlp.fc2.weight', 'proj_mlp.fc2.bias'}
    assert encoder['blocks.26.ln2.weight'].endswith('blocks.26.norm2.weight')
    assert encoder['patch_emb.weight'].endswith('patch_embed.linear.weight')


@pytest.mark.parametrize('suffix', ['.pt', '.safetensors'])
def test_loader_reads_only_named_vision_tensors(tmp_path, monkeypatch, suffix):
    sources = {'post_ln.bias': 'vision_encoder.encoder.model.visual.norm.bias'}
    monkeypatch.setattr('kestrel.models.siglip.weights.md3_vision_parameter_sources',
                        lambda n, include_projection: sources)
    tensors = {next(iter(sources.values())): torch.arange(3).float(),
               'text_model.unused': torch.zeros(9)}
    path = tmp_path / ('checkpoint' + suffix)
    if suffix == '.pt':
        torch.save(tensors, path)
    else:
        from safetensors.torch import save_file
        save_file(tensors, path)
    loaded = load_siglip_weights(path)
    assert set(loaded) == {'post_ln.bias'}
    torch.testing.assert_close(loaded['post_ln.bias'], torch.arange(3).bfloat16())


def test_unsupported_hopper_grid_fails_before_checkpoint_load(monkeypatch):
    monkeypatch.setattr('kestrel.models.siglip.runtime.get_device_capability', lambda device: (9, 0))
    monkeypatch.setattr('kestrel.models.siglip.runtime.get_device_sm_count', lambda device: 114)
    monkeypatch.setattr('kestrel.models.siglip.runtime.load_siglip_weights',
                        lambda path: pytest.fail('must reject before loading'))
    with pytest.raises(ValueError, match='132 SMs'):
        create_siglip_runtime(cfg('cuda:0', 'model_fp8.pt'))
