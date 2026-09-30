from __future__ import annotations

import numpy as np


MODEL_CONFIG = {
    "architectures": ["Dinov2Model"],
    "attention_probs_dropout_prob": 0.0,
    "drop_path_rate": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 384,
    "image_size": 518,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-6,
    "layerscale_value": 1.0,
    "mlp_ratio": 4,
    "model_type": "dinov2",
    "num_attention_heads": 6,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "patch_size": 14,
    "qkv_bias": True,
    "torch_dtype": "float32",
    "transformers_version": "4.32.0.dev0",
    "use_swiglu_ffn": False,
}

PROCESSOR_CONFIG = {
    "crop_size": {"height": 224, "width": 224},
    "do_center_crop": True,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "BitImageProcessor",
    "image_std": [0.229, 0.224, 0.225],
    "resample": 3,
    "rescale_factor": 1.0 / 255.0,
    "size": {"shortest_edge": 256},
}


def synthetic_rgb() -> np.ndarray:
    y, x = np.indices((73, 311), dtype=np.uint16)
    return np.stack(
        (
            (17 * x + 3 * y) % 256,
            np.bitwise_xor(x, 5 * y) % 256,
            np.where((x // 11 + y // 7) % 2, 255, 0),
        ),
        axis=-1,
    ).astype(np.uint8)
