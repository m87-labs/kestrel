"""Benchmark the fp Parakeet-TDT and the ternary export through plain PyTorch on CPU / MPS / CUDA.

Transcribes a directory of 16 kHz wavs one at a time with the reference greedy decoder (`ParakeetTdt.generate`),
features computed on CPU (torch.stft has no MPS kernel), and reports real-time factor, per-utterance latency and
peak resident memory. The hypotheses are written to `--out` as jsonl; WER comes from the thrush scorer over
that file.

  python scripts/bench_parakeet_ternary.py --model fp --device cpu --threads 4 \
      --bench-dir /path/to/devclean50 --out fp_cpu.jsonl
  python scripts/bench_parakeet_ternary.py --model ternary --export-dir /path/to/rl6-ternary-hf --device mps \
      --dtype fp16 --bench-dir /path/to/devclean50 --out ternary_mps.jsonl
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from kestrel.config import NATIVE_GEMM_THREAD_CAP, cpu_default_dtype, default_cpu_threads
from kestrel.models.parakeet_tdt.features import parakeet_features
from kestrel.models.parakeet_tdt.runtime import confine_to_cache_domain
from kestrel.models.parakeet_tdt.weights import MODEL_ID, load_parakeet_tdt

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # bytes on macOS, kilobytes on Linux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fp", "ternary"], required=True)
    ap.add_argument("--export-dir", default=None, help="thrush export with HF names (ternary)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto", choices=[*DTYPES, "auto"])
    ap.add_argument("--threads", type=int, default=0, help="0 applies the shipped CPU thread policy")
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    elif device.type == "cpu":
        # No --threads is the shipped policy: the kernels size and place their own pool, this process is
        # confined to the cache domain they chose, and torch keeps the small cap beside them.
        confine_to_cache_domain()
        torch.set_num_threads(default_cpu_threads(NATIVE_GEMM_THREAD_CAP))
    if args.dtype == "auto":  # the runtime's policy: bf16 on CUDA / native-bf16 CPUs, fp16 on MPS, fp32 elsewhere
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(device.type) or cpu_default_dtype()
        args.dtype = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
    else:
        dtype = DTYPES[args.dtype]

    t0 = time.time()
    if args.model == "ternary":
        checkpoint, label = args.export_dir, f"ternary {args.dtype}"
    else:
        checkpoint, label = MODEL_ID, f"fp {args.dtype}"
    loaded = load_parakeet_tdt(checkpoint, device=device, dtype=dtype)
    model, tokenizer = loaded.model, loaded.tokenizer
    load_s = time.time() - t0
    rss_after_load = peak_rss_mb()

    bench = Path(args.bench_dir)
    rows = [json.loads(l) for l in (bench / "refs.jsonl").read_text().splitlines() if l.strip()]

    feat_s = 0.0

    def transcribe(wav_path: Path) -> tuple[str, float, float]:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, sr
        t = time.time()
        features, mask = parakeet_features(torch.from_numpy(np.ascontiguousarray(audio)))  # CPU (no MPS stft kernel)
        features_s = time.time() - t
        t = time.time()
        with torch.inference_mode():
            out = model.generate(features.to(device=device, dtype=dtype), mask.to(device))
            ids = out.sequences[0].tolist()
        if device.type == "mps":
            torch.mps.synchronize()
        return tokenizer.decode(ids), features_s, time.time() - t

    transcribe(bench / "wav" / rows[0]["wav"])  # warm-up (allocations, kernel selection)
    hyps, lat = [], []
    wall = time.time()
    for r in rows:
        hyp, features_s, dt = transcribe(bench / "wav" / r["wav"])
        feat_s += features_s
        hyps.append(hyp)
        lat.append(dt)
    wall = time.time() - wall
    audio_s = sum(r["duration"] for r in rows)
    with open(args.out, "w") as f:
        for r, h in zip(rows, hyps):
            f.write(json.dumps({"id": r["id"], "hyp": h}) + "\n")
    print(json.dumps({
        "model": label, "device": args.device, "threads": torch.get_num_threads(), "n_utts": len(rows), "audio_s": round(audio_s, 1),
        "wall_s": round(wall, 2), "rtf_x": round(audio_s / wall, 2), "features_s": round(feat_s, 2), "latency_ms_median": round(1000 * float(np.median(lat)), 1),
        "latency_ms_p90": round(1000 * float(np.percentile(lat, 90)), 1), "load_s": round(load_s, 1),
        "peak_rss_mb_after_load": round(rss_after_load), "peak_rss_mb": round(peak_rss_mb()),
        "out": args.out,
    }))


if __name__ == "__main__":
    main()
