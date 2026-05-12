# Session-aware mamba radix cache

For this project, we have built a session-aware mamba radix cache for sglang.  By session aware, we mean the
assumption that requests within a given session will frequently re-use the most recent cache node, so we give
those nodes and their ancestors preference over other nodes in the cache.

We achieve this in the form of an additional "priority" key that precedes timestamp in the eviction heap.
* *High*: Nodes associated with the most recent turn in a session are retained over sessionless nodes
* *Normal*: Sessionless nodes are given *normal* priority
* *Low*: Stale session nodes are given reduced priority, evicted before all others.

## Design Decisions

* **Sessionless vs stale nodes**: We know we want to elevate the priority of the most recent nodes in a session, but how do we prioritize sessionless nodes vs stale session nodes.  The correct choice will ultimately depend on the use case, but it seems that the fact that a node was used in a session and now has been discarded seemed like sufficient signal that it should be deprioritized versus sessionless nodes.
* **Session status as absolute or modifier**: We opted for a rigorous, discrete hierarchy of priority, but there are scenarios where this is suboptimal, eg the pathological case of many sessions permanently stalled while a sessionless workload continues to operate would result in no cache usage whatsoever.  The rigid approach is simpler in theory and in practice, and the failure cases seem fairly exceptional compared to normal use cases, so we chose the rigid approach.
* **Session Independence**: One difficult design decision we encountered was whether to make priority an abstract concept in the cache, independent of sessionhood.  Rather than denote session membership in the nodes, we could simply expose priority in the api, allowing a higher level system map session ownership to priority.  We believe that ultimately code readability and maintainability makes this an unnecessary abstraction, the code can be refactored in the future as needed.
* **Dereference fully evicted nodes**: Does it make sense for a node to be marked with elevated priority if it has been evicted from the cache?  We believe that the code is simpler by separating the logic of priority from eviction, and so in this implementation a node being evicted from the cache does not release its priority for retention.  There are extreme cases where a sessionless request could re-populate the identical cache node, in which case the priority claim would still be relevant.  This isn't likely in practice, but this implementation seems cleaner in theory.


# Session Cache Pressure Harness

This directory contains a checked-in harness for replaying a synthetic multi-turn
session workload against a running SGLang server.

The private notes in this directory remain ignored by git.

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
