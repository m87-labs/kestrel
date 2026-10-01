"""Single-pass image tokens from the installed SigLIP encoder family."""
from collections.abc import Mapping
from concurrent.futures import Future
from contextlib import nullcontext

import torch

from kestrel.device import empty_cache, get_device_capability, get_device_sm_count, resolve_device
from kestrel.models.moondream.config import VisionConfig
from kestrel.runtime import ExecutionShape
from .preprocessing import preprocess_image
from .weights import load_siglip_weights


class SiglipRuntime:
    execution_shape = ExecutionShape.SINGLE_PASS
    # One request can contain 1..13 independent images in pixel_values.
    batch_capacity = 1

    def __init__(self, cfg, *, executable, compute_stream=None):
        self.model_name = cfg.model
        self.device = resolve_device(cfg.resolved_device())
        self.dtype = cfg.resolved_dtype()
        self.primary_stream = self.compute_stream = compute_stream
        self.executable = executable
        self._shutdown = False

    def tasks(self):
        return ("embed",)

    def preprocess_image_async(self, image):
        future = Future()
        try:
            future.set_result(preprocess_image(image))
        except Exception as exc:
            future.set_exception(exc)
        return future

    @torch.inference_mode()
    def forward(self, task, inputs):
        if self._shutdown:
            raise RuntimeError("SiglipRuntime is shut down")
        if task != "embed" or len(inputs) != 1:
            raise ValueError("SigLIP serves one embed request per forward")
        request = inputs[0]
        if not isinstance(request, Mapping):
            raise TypeError("embed inputs must be a mapping")
        if set(request) - {"image", "pixel_values"}:
            raise ValueError("SigLIP accepts image or pixel_values")
        if ("image" in request) == ("pixel_values" in request):
            raise ValueError("embed requires exactly one of image or pixel_values")
        pixels = (preprocess_image(request["image"]) if "image" in request
                  else request["pixel_values"])
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("pixel_values must be a torch.Tensor")
        if (pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 378, 378)
                or not 1 <= pixels.shape[0] <= 13):
            raise ValueError("pixel_values must have shape [N,3,378,378], N in 1..13")
        if pixels.dtype is not torch.uint8:
            raise TypeError("pixel_values must be raw uint8 pixels")
        if pixels.device.type == "cpu":
            pixels = pixels.contiguous().to(self.device)
        elif pixels.device != self.device or not pixels.is_contiguous():
            raise ValueError("GPU pixel_values must be contiguous on the model device")
        output = self.executable.encode_crops(pixels)
        if (not isinstance(output, torch.Tensor) or
                tuple(output.shape) != (pixels.shape[0], 729, 1152)
                or output.dtype is not torch.bfloat16 or output.device != self.device):
            raise RuntimeError("SigLIP must return BF16 [N,729,1152] on the model device")
        return ({"last_hidden_state": output},)

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        try:
            self.executable.close()
        finally:
            self.executable = None
            empty_cache(self.device)


def create_siglip_runtime(cfg, *, compute_stream=None, kv_pool=None, max_lora_rank=None):
    del kv_pool, max_lora_rank
    device, dtype = resolve_device(cfg.resolved_device()), cfg.resolved_dtype()
    if (device.type != "cuda" or dtype is not torch.bfloat16
            or tuple(get_device_capability(device)) != (9, 0)):
        raise ValueError("SigLIP currently requires Hopper CUDA and BF16")
    if cfg.model_path is None:
        raise ValueError("SigLIP requires model_path to the qualified Moondream 3 checkpoint")
    if get_device_sm_count(device) != 132:
        raise ValueError("SigLIP currently requires an H100 with 132 SMs")
    state_dict = load_siglip_weights(cfg.model_path)
    from kestrel_kernels.megakernel.siglip import SiglipTokenEncoder
    with torch.cuda.device(device), (
        torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
    ):
        executable = SiglipTokenEncoder(state_dict=state_dict, config=VisionConfig(),
                                        device=device, dtype=dtype)
    return SiglipRuntime(cfg, executable=executable, compute_stream=compute_stream)
