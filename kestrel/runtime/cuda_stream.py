"""Owned, non-pooled CUDA streams for independently retired graph captures."""

import ctypes
from functools import cache
import sys

import torch


@cache
def _stream_api():
    driver = (ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32"
              else ctypes.CDLL("libcuda.so.1"))
    create = driver.cuStreamCreate
    create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    create.restype = ctypes.c_int
    destroy = driver.cuStreamDestroy_v2
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    return create, destroy


class OwnedCudaStream:
    """Keep a unique capture stream alive until its graph has been reset."""

    def __init__(self, device):
        self.device = torch.device(device)
        self._handle = ctypes.c_void_p()
        create, self._destroy = _stream_api()
        with torch.cuda.device(self.device):
            # Establish the runtime's primary context even before any allocation.
            torch.cuda.mem_get_info(self.device)
            status = create(ctypes.byref(self._handle), 1)  # CU_STREAM_NON_BLOCKING
            if status:
                raise RuntimeError(f"cuStreamCreate failed with CUDA driver status {status}")
            try:
                self.stream = torch.cuda.ExternalStream(self._handle.value, device=self.device)
            except BaseException:
                self._destroy(self._handle)
                self._handle = ctypes.c_void_p()
                raise

    def close(self):
        if self._handle.value is None:
            return
        with torch.cuda.device(self.device):
            self.stream.synchronize()
            status = self._destroy(self._handle)
            if status:
                raise RuntimeError(f"cuStreamDestroy failed with CUDA driver status {status}")
        self._handle = ctypes.c_void_p()
