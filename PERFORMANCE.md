# Performance

Throughput and latency for the `query` skill, measured on the [ChartQA](https://huggingface.co/datasets/vikhyatk/chartqa) test split with prefix caching enabled.

- **Direct** - the model generates a short answer (~3 output tokens per request)
- **CoT** (Chain-of-Thought) - the model reasons step-by-step before answering (~30 output tokens per request), enabled via `reasoning=True`

## H100

### Moondream 2

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 36.92 | 43.90 | 12.59 | 132.07 |
| 4 | 42.17 | 87.10 | 24.32 | 166.78 |
| 16 | 40.73 | 295.48 | 32.80 | 396.49 |
| 64 | 43.63 | 1159.04 | 37.14 | 1090.10 |

### Moondream 3

| Batch Size | Direct (req/s) | Direct P50 (ms) | CoT (req/s) | CoT P50 (ms) |
|-----------|---------------|----------------|-------------|--------------|
| 1 | 24.56 | 71.34 | 9.11 | 186.07 |
| 4 | 35.49 | 103.57 | 16.84 | 249.37 |
| 16 | 37.94 | 326.50 | 22.69 | 574.26 |
| 64 | 37.52 | 1289.38 | 29.88 | 1364.20 |
