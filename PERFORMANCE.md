# Performance

Throughput and latency for the `query` skill, measured on the [ChartQA](https://huggingface.co/datasets/vikhyatk/chartqa) test split with prefix caching enabled.

- **Direct** - the model generates a short answer (~3 output tokens per request)
- **CoT** (Chain-of-Thought) - the model reasons step-by-step before answering (~30 output tokens per request), enabled via `reasoning=True`

## H100

### Moondream 2

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 35.54 | 46.33 | 9.32 | 180.77 |
| 4 | 49.89 | 74.24 | 21.45 | 194.64 |
| 16 | 59.33 | 145.46 | 37.77 | 353.16 |
| 64 | 62.83 | 353.91 | 48.44 | 910.44 |

### Moondream 3

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 29.47 | 62.16 | 8.01 | 213.77 |
| 4 | 40.39 | 91.37 | 16.19 | 262.82 |
| 16 | 49.88 | 184.94 | 28.11 | 485.78 |
| 64 | 58.01 | 409.92 | 39.30 | 1185.06 |

## L40S

### Moondream 2

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 17.15 | 100.25 | 5.19 | 327.61 |
| 4 | 17.25 | 215.47 | 11.20 | 388.62 |
| 16 | 20.35 | 479.49 | 14.89 | 960.33 |
| 64 | 21.34 | 1271.78 | 17.07 | 2844.01 |

### Moondream 3

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 14.83 | 121.19 | 4.48 | 390.78 |
| 4 | 14.90 | 267.93 | 6.81 | 642.23 |
| 16 | 19.57 | 512.59 | 10.47 | 1336.79 |
| 64 | 18.81 | 1538.96 | 14.35 | 3425.99 |
## A100 80GB

Moondream 3 is not yet supported on Ampere GPUs and currently requires `sm89+`, so Ampere results below reflect Moondream 2 only.

### Moondream 2

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 16.27 | 104.45 | 5.59 | 305.25 |
| 4 | 18.92 | 204.75 | 11.32 | 368.55 |
| 16 | 22.09 | 439.39 | 17.08 | 833.40 |
| 64 | 21.36 | 1275.38 | 15.77 | 2947.67 |
## A10

Moondream 3 is not yet supported on Ampere GPUs and currently requires `sm89+`, so Ampere results below reflect Moondream 2 only.

### Moondream 2

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 7.60 | 223.18 | 2.96 | 607.39 |
| 4 | 9.00 | 491.78 | 5.85 | 766.23 |
| 16 | 8.44 | 1164.98 | 6.72 | 1844.57 |
| 64 | 6.83 | 1872.26 | 5.36 | 2059.68 |
## L4

### Moondream 2

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 5.75 | 302.01 | 2.04 | 873.47 |
| 4 | 6.62 | 646.75 | 4.10 | 1091.00 |
| 16 | 7.61 | 1299.17 | 5.63 | 2208.90 |
| 64 | 5.59 | 2282.66 | 4.08 | 2711.66 |

### Moondream 3

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 4.95 | 358.01 | 1.86 | 961.24 |
| 4 | 6.00 | 701.51 | 2.78 | 1597.44 |
| 16 | 6.27 | 1646.85 | 4.11 | 3121.25 |
| 64 | 4.85 | 2670.47 | 3.01 | 3680.44 |
## Jetson AGX Orin (32GB)

### Moondream 2

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 3.20 | 543.31 | 1.21 | 1480.97 |
| 4 | 3.66 | 1219.86 | 2.34 | 1940.81 |
