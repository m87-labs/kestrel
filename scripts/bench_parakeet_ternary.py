"""Benchmark the fp Parakeet-TDT and the ternary export through plain PyTorch on CPU / MPS / CUDA.

Transcribes a directory of 16 kHz wavs one at a time with the reference greedy decoder (`ParakeetTdt.generate`),
features computed on CPU (torch.stft has no MPS kernel), and reports real-time factor, per-utterance latency, peak
resident memory and a quick normalized WER against refs.jsonl (lowercase, punctuation stripped; the official
numbers come from the thrush scorer on the written hypotheses).

  python scripts/bench_parakeet_ternary.py --model fp --device cpu --threads 4 \
      --bench-dir /path/to/devclean50 --out fp_cpu.jsonl
  python scripts/bench_parakeet_ternary.py --model ternary --export-dir /path/to/rl6-ternary-hf --device mps \
      --dtype fp16 --mode dense --bench-dir /path/to/devclean50 --out ternary_mps.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # bytes on macOS, kilobytes on Linux


def quick_wer(refs: list[str], hyps: list[str]) -> float | None:
    try:
        import jiwer
    except ImportError:
        return None

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s.lower())).strip()

    r = [norm(x) for x in refs]
    h = [norm(x) for x in hyps]
    keep = [i for i, x in enumerate(r) if x]
    return 100.0 * jiwer.wer([r[i] for i in keep], [h[i] for i in keep])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fp", "ternary"], required=True)
    ap.add_argument("--export-dir", default=None, help="thrush export with HF names (ternary)")
    ap.add_argument("--checkpoint", default=None, help="fp checkpoint (default: the pinned nvidia/parakeet-tdt-0.6b-v3)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto", choices=[*DTYPES, "auto"])
    ap.add_argument("--mode", default="auto", choices=["int8", "dense", "packed", "jit", "vnni", "auto"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--legacy-conv", action="store_true", help="keep nn.Conv1d for the depthwise conv on CPU/MPS (A/B)")
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if args.dtype == "auto":  # the runtime's policy: bf16 on CUDA / native-bf16 CPUs, fp16 on MPS, fp32 elsewhere
        from kestrel.config import cpu_default_dtype

        dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(device.type) or cpu_default_dtype()
        args.dtype = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
    else:
        dtype = DTYPES[args.dtype]
    if args.legacy_conv:
        import kestrel.models.parakeet_tdt.model as _pm

        _pm.DEPTHWISE_LINEAR_LAYOUT_DEVICES = frozenset()
    from kestrel.models.parakeet_tdt.features import parakeet_features

    t0 = time.time()
    if args.model == "fp":
        from kestrel.models.parakeet_tdt.weights import load_parakeet_tdt

        from kestrel.models.parakeet_tdt.weights import MODEL_ID

        loaded = load_parakeet_tdt(args.checkpoint or MODEL_ID, device=device, dtype=dtype)
        model, tokenizer = loaded.model, loaded.tokenizer
        label = f"fp {args.dtype}"
    else:
        from kestrel.models.parakeet_tdt.weights import load_parakeet_tdt_ternary

        loaded = load_parakeet_tdt_ternary(args.export_dir, device=device, dtype=dtype, mode=args.mode)
        model, tokenizer = loaded.model, loaded.tokenizer
        label = f"ternary {args.mode} {args.dtype}"
    load_s = time.time() - t0
    rss_after_load = peak_rss_mb()

    bench = Path(args.bench_dir)
    rows = [json.loads(l) for l in (bench / "refs.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    feat_s = [0.0]

    def transcribe(wav_path: Path) -> tuple[str, float]:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, sr
        t = time.time()
        features, mask = parakeet_features(torch.from_numpy(np.ascontiguousarray(audio)))  # CPU (no MPS stft kernel)
        feat_s[0] += time.time() - t
        t = time.time()
        with torch.inference_mode():
            out = model.generate(features.to(device=device, dtype=dtype), mask.to(device))
            ids = out.sequences[0].tolist()
        if device.type == "mps":
            torch.mps.synchronize()
        return tokenizer.decode(ids), time.time() - t

    transcribe(bench / "wav" / rows[0]["wav"])  # warm-up (allocations, kernel selection)
    feat_s[0] = 0.0
    hyps, lat = [], []
    wall = time.time()
    for r in rows:
        hyp, dt = transcribe(bench / "wav" / r["wav"])
        hyps.append(hyp)
        lat.append(dt)
    wall = time.time() - wall
    audio_s = sum(r["duration"] for r in rows)
    with open(args.out, "w") as f:
        for r, h in zip(rows, hyps):
            f.write(json.dumps({"id": r["id"], "hyp": h}) + "\n")
    wer = quick_wer([r["text"] for r in rows], hyps)
    print(json.dumps({
        "model": label, "device": args.device, "threads": torch.get_num_threads(), "n_utts": len(rows), "audio_s": round(audio_s, 1),
        "wall_s": round(wall, 2), "rtf_x": round(audio_s / wall, 2), "features_s": round(feat_s[0], 2), "latency_ms_median": round(1000 * float(np.median(lat)), 1),
        "latency_ms_p90": round(1000 * float(np.percentile(lat, 90)), 1), "load_s": round(load_s, 1),
        "peak_rss_mb_after_load": round(rss_after_load), "peak_rss_mb": round(peak_rss_mb()), "quick_wer": None if wer is None else round(wer, 2),
        "out": args.out,
    }))


if __name__ == "__main__":
    main()
