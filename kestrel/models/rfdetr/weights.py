"""Load tensor-only inference checkpoints without upstream model construction."""
from pathlib import Path
from argparse import Namespace

import torch

# Asset names follow upstream rfdetr/assets/model_weights.py.
# Large means the current single-P4 2026 model.
_URLS = {
    'nano': 'nano_coco/checkpoint_best_regular.pth',
    'small': 'small_coco/checkpoint_best_regular.pth',
    'medium': 'medium_coco/checkpoint_best_regular.pth',
    'base': 'rf-detr-base-coco.pth',
    'large': 'rf-detr-large-2026.pth',
}


def load_weights(variant, path=None):
    if path is None:
        if variant not in _URLS:
            raise ValueError(f'rfdetr-{variant} requires model_path to its released checkpoint')
        with torch.serialization.safe_globals([Namespace]):
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://storage.googleapis.com/rfdetr/' + _URLS[variant],
                map_location='cpu', file_name=f'kestrel-rfdetr-{variant}.pth',
                weights_only=True)
    else:
        with torch.serialization.safe_globals([Namespace]):
            checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('RF-DETR checkpoint must be a tensor state dictionary or model mapping')
    state = checkpoint.get('model', checkpoint)
    if not isinstance(state, dict) or not state or any(
        not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError('RF-DETR checkpoint must contain a tensor state dictionary')
    return state
