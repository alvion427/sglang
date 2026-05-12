# Session Cache Pressure Harness

This directory contains a checked-in harness for replaying a synthetic multi-turn
session workload against a running SGLang server.

The private assignment notes in this directory remain ignored by git.

## Basic Usage

Use an existing server:

```bash
python mamba-radix-cache/session_cache_pressure_harness.py \
  --base-url http://127.0.0.1:30000 \
  --model default
```

Let the harness start a server:

```bash
python mamba-radix-cache/session_cache_pressure_harness.py \
  --start-server \
  --model-path Qwen/Qwen3.5-4B \
  --server-arg=--enable-hierarchical-cache \
  --server-arg=--mem-fraction-static --server-arg=0.35 \
  --server-arg=--max-total-tokens --server-arg=32768 \
  --server-arg=--hicache-ratio --server-arg=3.0 \
  --server-arg=--max-mamba-cache-size --server-arg=16 \
  --server-arg=--max-running-requests --server-arg=4 \
  --server-arg=--disable-cuda-graph \
  --server-arg=--attention-backend --server-arg=triton
```

For HiMamba radix cache tests, always launch with
`--server-arg=--enable-hierarchical-cache`. Without it, SGLang uses
`MambaRadixCache`, so the run does not exercise `HiMambaRadixCache`.

To compare old-LRU behavior with session-aware behavior, run twice. The harness
can pass the intended debug environment variables through to the launched server.
Use a fresh server process for each side of an A/B comparison; otherwise the
second run inherits cache state from the first run.

```bash
python mamba-radix-cache/session_cache_pressure_harness.py \
  --start-server \
  --model-path ... \
  --server-arg=--enable-hierarchical-cache \
  --server-arg=--max-total-tokens --server-arg=32768 \
  --server-arg=--hicache-ratio --server-arg=3.0 \
  --server-arg=--max-mamba-cache-size --server-arg=16 \
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

The default workload is sized as a local A/B comparison: 16 sessions, 5 turns,
short generated responses, and a small stateless background load. With the
server arguments above, KV capacity is large enough that latest-session loss is
primarily a Mamba-retention policy signal rather than a hard KV-capacity limit.
The default tool-sleep distribution is intentionally skewed toward short or zero
sleeps with occasional long pauses: `--tool-sleep-zero-probability 0.5`,
`--tool-sleep-power 4`, and `--tool-sleep-ms 20000`.
