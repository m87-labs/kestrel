"""Inference geometry for the seven qualified single-P4 COCO detectors."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RFDetrConfig:
    variant: str
    resolution: int
    patch_size: int
    positional_encoding_size: int
    dec_layers: int
    num_windows: int = 2
    hidden_dim: int = 256
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    out_feature_indexes: tuple[int, ...] = (3, 6, 9, 12)
    num_queries: int = 300
    num_classes: int = 90  # Checkpoints have 91 logit slots; IDs retain COCO gaps.
    num_channels: int = 3
    projector_scale: tuple[str, ...] = ('P4',)
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    two_stage: bool = True


CONFIGS = {
    'nano': RFDetrConfig('nano', 384, 16, 24, 2),
    'small': RFDetrConfig('small', 512, 16, 32, 3),
    'medium': RFDetrConfig('medium', 576, 16, 36, 4),
    'base': RFDetrConfig('base', 560, 14, 37, 3, num_windows=4,
                         out_feature_indexes=(2, 5, 8, 11)),
    'large': RFDetrConfig('large', 704, 16, 44, 4),
    'xlarge': RFDetrConfig('xlarge', 700, 20, 35, 5, num_windows=1,
                           hidden_dim=512, sa_nheads=16, ca_nheads=32, dec_n_points=4),
    '2xlarge': RFDetrConfig('2xlarge', 880, 20, 44, 5,
                            hidden_dim=512, sa_nheads=16, ca_nheads=32, dec_n_points=4),
}
