# Session Cache Pressure Harness

This directory contains a checked-in harness for replaying a synthetic multi-turn
session workload against a running SGLang server.

The private assignment notes in this directory remain ignored by git.

## Basic Usage

Use an existing server:

```bash
python mamba-radix-cache/session_cache_pressure_harness.py \
  --base-url http://127.0.0.1:30000 \
  --model default \
  --sessions 64 \
  --turns 8 \
  --common-headers 4 \
  --concurrency 32
```

Let the harness start a server:

```bash
python mamba-radix-cache/session_cache_pressure_harness.py \
  --start-server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --server-arg=--tp --server-arg=4 \
  --server-arg=--mem-fraction-static --server-arg=0.70 \
  --sessions 64 \
  --turns 8
```

To compare old-LRU behavior with session-aware behavior, run twice. The harness
can pass the intended debug environment variables through to the launched server:

```bash
python mamba-radix-cache/session_cache_pressure_harness.py \
  --start-server \
  --model-path ... \
  --old-lru-mode \
  --warn-on-latest-evict
```

The server-side env vars are expected to be:

- `SGLANG_HIMAMBA_DISABLE_SESSION_RETENTION_PRIORITY=1`
- `SGLANG_HIMAMBA_WARN_ON_LATEST_RETENTION_EVICT=1`

The harness writes:

- a JSON summary with latency, token, cache-hit, and warning counts
- a JSONL file with per-request details
- an optional server log if it launched the server
