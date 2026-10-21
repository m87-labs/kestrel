"""Exclusive CUDA capture-stream leases, reused only after graph retirement."""

import ctypes
from functools import cache
import sys
import threading

import torch


_idle_streams = {}
_stream_lock = threading.Lock()


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
    """Keep a capture stream exclusive until its graph has been reset.

    PyTorch 2.11 caches cuBLAS workspaces by handle and stream, even after
    cudaStreamDestroy. Retiring 40 fresh H100 streams retained 1.25 GiB.
    Reuse only synchronized, retired streams so workspace use is bounded by
    peak live captures, without clearing workspaces owned by another graph.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self._handle = ctypes.c_void_p()
        create, self._destroy = _stream_api()
        with torch.cuda.device(self.device):
            # Establish the runtime's primary context even before any allocation.
            torch.cuda.mem_get_info(self.device)
            self._device_index = torch.cuda.current_device()
            self.device = torch.device("cuda", self._device_index)
            with _stream_lock:
                idle = _idle_streams.setdefault(self._device_index, [])
                reused = idle.pop() if idle else None
            if reused is not None:
                handle, self.stream = reused
                self._handle = ctypes.c_void_p(handle)
                return
            status = create(ctypes.byref(self._handle), 1)  # CU_STREAM_NON_BLOCKING
            if status:
                raise RuntimeError(f"cuStreamCreate failed with CUDA driver status {status}")
            try:
                self.stream = torch.cuda.ExternalStream(self._handle.value, device=self.device)
            except BaseException:
                self._destroy(self._handle)
                self._handle = ctypes.c_void_p()
                raise

    def close(self, *, reuse=True):
        if self._handle.value is None:
            return
        with torch.cuda.device(self.device):
            self.stream.synchronize()
            with _stream_lock:
                if self._handle.value is None:
                    return
                if reuse:
                    _idle_streams.setdefault(self._device_index, []).append(
                        (self._handle.value, self.stream))
                else:
                    status = self._destroy(self._handle)
                    if status:
                        raise RuntimeError(f"cuStreamDestroy failed with CUDA driver status {status}")
                self._handle = ctypes.c_void_p()
