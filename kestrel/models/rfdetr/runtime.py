"""Batch-one fixed-vocabulary detection through the shipped Hopper executable."""
from collections.abc import Mapping
from concurrent.futures import Future
from contextlib import nullcontext
import math

import torch

from kestrel.device import get_device_capability, resolve_device
from kestrel.runtime import ExecutionShape
from .config import CONFIGS
from .labels import COCO_CLASSES
from .preprocessing import preprocess_image
from .weights import load_weights


def postprocess(boxes, logits, *, threshold=.5, max_objects=300):
    """CPU sigmoid/top-query-class selection, without NMS or class reindexing."""
    if boxes.device.type != 'cpu' or logits.device.type != 'cpu':
        raise ValueError('postprocessing requires owned CPU outputs')
    if tuple(boxes.shape) != (300, 4) or tuple(logits.shape) != (300, 91):
        raise ValueError('RF-DETR COCO outputs must be [300,4] boxes and [300,91] logits')
    boxes, logits = boxes.float(), logits.float()
    if not torch.isfinite(boxes).all() or not torch.isfinite(logits).all():
        raise RuntimeError('RF-DETR produced non-finite outputs')
    scores, indexes = logits.sigmoid().flatten().topk(300)
    objects = []
    for score, index in zip(scores.tolist(), indexes.tolist()):
        if score <= threshold or len(objects) == max_objects:
            break
        query, class_id = divmod(index, 91)
        cx, cy, width, height = boxes[query].tolist()
        objects.append(dict(x_min=max(0., min(1., cx - width / 2)),
                            y_min=max(0., min(1., cy - height / 2)),
                            x_max=max(0., min(1., cx + width / 2)),
                            y_max=max(0., min(1., cy + height / 2)),
                            score=score, class_id=class_id,
                            label=COCO_CLASSES.get(class_id, '')))
    return {'objects': objects}


class RFDetrRuntime:
    execution_shape = ExecutionShape.SINGLE_PASS
    batch_capacity = 1

    def __init__(self, *, name, config, executable, device, compute_stream=None):
        self.model_name = name
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16
        self.primary_stream = compute_stream
        self.compute_stream = compute_stream
        self.executable = executable
        self._shutdown = False

    def tasks(self):
        return ('detect',)

    def preprocess_image_async(self, image):
        future = Future()
        try:
            future.set_result(preprocess_image(image, self.config.resolution))
        except Exception as exc:
            future.set_exception(exc)
        return future

    @torch.inference_mode()
    def forward(self, task, inputs):
        if self._shutdown:
            raise RuntimeError('RFDetrRuntime is shut down')
        if task != 'detect' or len(inputs) != 1:
            raise ValueError('RF-DETR serves one detect request per forward')
        request = inputs[0]
        if not isinstance(request, Mapping):
            raise TypeError('detect inputs must be a mapping')
        if set(request) - {'image', 'pixel_values', 'threshold', 'max_objects'}:
            raise ValueError('RF-DETR accepts image or pixel_values, threshold and max_objects; it has a fixed COCO vocabulary')
        if ('image' in request) == ('pixel_values' in request):
            raise ValueError('detect requires exactly one of image or pixel_values')
        threshold = request.get('threshold', .5)
        limit = request.get('max_objects', 300)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must lie in [0, 1]')
        if type(limit) is not int or not 1 <= limit <= 300:
            raise ValueError('max_objects must be an integer in [1, 300]')
        pixels = (preprocess_image(request['image'], self.config.resolution)
                  if 'image' in request else request['pixel_values'])
        expected = (1, 3, self.config.resolution, self.config.resolution)
        if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != expected or not pixels.is_floating_point():
            raise ValueError(f'pixel_values must be a floating-point tensor with shape {expected}')
        # CPU casting avoids a separate GPU conversion kernel. Already prepared
        # GPU pixels must use the executable's BF16 dtype and contiguous layout.
        if pixels.device.type == 'cpu':
            pixels = pixels.to(dtype=self.dtype).contiguous()
            if not torch.isfinite(pixels).all():
                raise ValueError('pixel_values must be finite after BF16 conversion')
            pixels = pixels.to(self.device)
        elif pixels.device != self.device or pixels.dtype != self.dtype or not pixels.is_contiguous():
            raise ValueError('GPU pixel_values must be contiguous BF16 on the model device')
        with self.executable.borrow_outputs(pixels) as (boxes, logits):
            # Complete boundary readback while the executable owns its buffers.
            # All postprocessing arithmetic follows on CPU, outside GPU forward.
            cpu_boxes, cpu_logits = boxes.cpu(), logits.cpu()
            result = postprocess(cpu_boxes[:self.config.num_queries, :4],
                                 cpu_logits[:self.config.num_queries, :91],
                                 threshold=threshold, max_objects=limit)
        return (result,)

    def shutdown(self):
        if not self._shutdown:
            self._shutdown = True
            self.executable.close()


def create_rfdetr_runtime(cfg, *, compute_stream=None, kv_pool=None, max_lora_rank=None):
    del kv_pool, max_lora_rank
    device, dtype = resolve_device(cfg.resolved_device()), cfg.resolved_dtype()
    if device.type != 'cuda' or dtype != torch.bfloat16 or tuple(get_device_capability(device)) != (9, 0):
        raise ValueError('RF-DETR currently requires Hopper CUDA and BF16')
    variant = cfg.model.removeprefix('rfdetr-')
    config = CONFIGS[variant]
    state = load_weights(variant, cfg.model_path)
    from kestrel_kernels.megakernel.rfdetr import RFDetrMegakernelDetector
    with torch.cuda.device(device), (torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()):
        executable = RFDetrMegakernelDetector(state, config, device=device, dtype=dtype)
    return RFDetrRuntime(name=cfg.model, config=config, executable=executable,
                         device=device, compute_stream=compute_stream)
