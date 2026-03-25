# Jetson Setup Guide

Kestrel supports NVIDIA Jetson Orin (AGX Orin, Orin NX) with JetPack 6.0, 6.1, or 6.2.

## Prerequisites

- Jetson Orin with JetPack 6.x flashed
- Python 3.10
- CUDA runtime (included with JetPack)

## Install PyTorch

Jetson requires NVIDIA's custom PyTorch wheels. Install the version matching your JetPack release.

**JetPack 6.1 / 6.2:**
```bash
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

**JetPack 6.0:**
```bash
pip install https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
```

If using `uv`, set `UV_SKIP_WHEEL_FILENAME_CHECK=1` — NVIDIA's wheel filenames
contain a build identifier that doesn't match the internal metadata.

## Install Kestrel

After PyTorch is installed, install Kestrel itself:

```bash
# pip
pip install "numpy<2" kestrel

# uv
uv pip install "numpy<2" kestrel
```

The current Jetson PyTorch wheels are still built against NumPy 1.x, so
installing `numpy<2` avoids the compatibility warning emitted by `import torch`.

## Set `LD_LIBRARY_PATH`

NVIDIA's Jetson PyTorch wheel needs the JetPack CUDA libraries on the library
path. If `import torch` fails with errors about missing `libnvToolsExt.so.1`,
`libcublas.so`, or `libcupti.so`, set `LD_LIBRARY_PATH` to include the CUDA
library directory:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
```

If you still see errors about `libcupti.so` or `libnvToolsExt.so`, you may need
to install additional CUDA packages:

```bash
sudo apt install cuda-cupti-12-6 libnvtoolsext1
```

If `import torch` fails with `libcusparseLt.so.0`, install cuSPARSELt for your
JetPack/CUDA version and add its `lib` directory to `LD_LIBRARY_PATH` as well.
On some pre-provisioned Jetson machines this may live in a user-local path
rather than under `/usr/local/cuda`.

You may want to add this to your shell profile (`~/.bashrc` or similar) so it
persists across sessions.

## Verify

```bash
python3 -c "import torch; print(torch.__version__); import kestrel; print('kestrel OK')"
```

## Benchmarking

For reproducible benchmark numbers on Orin, make sure the board is in its
uncapped power mode and pin clocks before running benchmarks:

```bash
sudo nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --show
```

If `nvpmodel -q` does not report `MAXN`, switch the board to its MAXN mode
before benchmarking.

## Notes

- Only Python 3.10 is supported on Jetson (matching NVIDIA's torch wheel).
- `nvcc` is not required at runtime.
- Triton (the compiler, not Triton Inference Server) is optional and not available on aarch64 — Kestrel automatically falls back to pure PyTorch implementations where needed.
