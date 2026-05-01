# Jetson Setup Guide

Kestrel supports two Jetson families on aarch64 Linux:

- **Jetson AGX Orin / Orin NX** (sm_87) on **JetPack 6** (6.0, 6.1, 6.2).
- **Jetson AGX Thor** (sm_110) on **JetPack 7**.

The kestrel-kernels wheel is a single multi-CUDA aarch64 build that
covers both — a cu12 codepath dispatches on JetPack 6 (Orin) and a
cu13 codepath dispatches on JetPack 7 (Thor). The `pip install`
command is the same on either device.

---

## Jetson AGX Orin / Orin NX (JetPack 6)

### Prerequisites

- Jetson Orin device with JetPack 6.x flashed.
- Python 3.10 (matches NVIDIA's JetPack 6 PyTorch wheel).
- CUDA runtime included with JetPack.

### Install PyTorch

JetPack 6 ships an old CUDA 12.x and requires NVIDIA's custom PyTorch
wheel (PyPI's stock aarch64 torch wheels target newer CUDA):

**JetPack 6.1 / 6.2:**
```bash
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

**JetPack 6.0:**
```bash
pip install https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
```

If using `uv`, set `UV_SKIP_WHEEL_FILENAME_CHECK=1` — NVIDIA's wheel
filenames contain a build identifier that doesn't match the internal
metadata.

### Install Kestrel

```bash
pip install "numpy<2" kestrel
```

JetPack 6's PyTorch wheel is built against NumPy 1.x, so installing
`numpy<2` avoids the import-time compatibility warning.

### Set `LD_LIBRARY_PATH`

JetPack 6's PyTorch wheel loads CUDA libraries from the system
JetPack install. If `import torch` fails with errors about missing
`libnvToolsExt.so.1`, `libcublas.so`, or `libcupti.so`, add the
JetPack CUDA library directories to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
```

If you still see errors about `libcupti.so` or `libnvToolsExt.so`,
install the missing CUDA packages:

```bash
sudo apt install cuda-cupti-12-6 libnvtoolsext1
```

If `import torch` fails with `libcusparseLt.so.0`, install cuSPARSELt
for your JetPack/CUDA version and add its `lib` directory to
`LD_LIBRARY_PATH` as well. On some pre-provisioned Jetson machines
this lives in a user-local path rather than under `/usr/local/cuda`.

You may want to add the `LD_LIBRARY_PATH` export to your shell
profile (`~/.bashrc` or similar) so it persists across sessions.

---

## Jetson AGX Thor (JetPack 7)

### Prerequisites

- Jetson Thor device with JetPack 7 flashed.
- Python 3.12 (matches JetPack 7's system Python).

### Install Kestrel

JetPack 7 ships CUDA 13, which the standard PyPI PyTorch aarch64
wheel targets — no custom NVIDIA wheel needed:

```bash
pip install kestrel
```

This pulls in PyTorch 2.x for aarch64 along with the `nvidia-*-cu13`
runtime packages it depends on.

### Set `LD_LIBRARY_PATH`

PyTorch on Thor loads CUDA libraries from the pip-installed
`nvidia-*-cu13` packages and `nvpl` (NVIDIA Performance Libraries —
BLAS / LAPACK / FFT for aarch64). Both live under your venv's
site-packages, not `/usr/local/cuda`. Point `LD_LIBRARY_PATH` at the
venv directories:

```bash
SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIBS=$(find "$SP" -maxdepth 4 -type d -name lib 2>/dev/null \
       | grep -E '/(nvidia|nvpl)/' | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="$LIBS:$LD_LIBRARY_PATH"
```

(Add the export to your shell profile so it persists across sessions.)

If `import torch` then fails with a `libcudnn.so.9` or
`libnvpl_lapack_lp64_gomp.so.0` error, double-check that the `find`
command picked up both `nvidia/cudnn/lib/` and `nvpl/lib/`.

---

## Verify

On either device:

```bash
python3 -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
import kestrel
print('kestrel OK')
"
```

Expect `cuda True` and `device NVIDIA Thor` (or `Orin`).

---

## Notes

- **Python**: JetPack 6 (Orin) is Python 3.10 — matches NVIDIA's torch
  wheel. JetPack 7 (Thor) is Python 3.12 — matches the system Python
  and the standard PyPI torch wheel.
- **Triton** (the compiler, not Triton Inference Server) isn't packaged
  for aarch64. Kestrel detects this at import and proceeds without
  LoRA support; base inference for Moondream 2 / Moondream 3 is
  unaffected.
- **`nvcc`** is not required at runtime on either device.
