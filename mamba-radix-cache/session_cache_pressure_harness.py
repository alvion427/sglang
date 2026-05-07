#!/usr/bin/env python3
"""Replay synthetic session trajectories against an SGLang server.

The workload is designed to create cache pressure with realistic multi-turn
session shape: shared system prompts, per-session state, incremental turns, and
background stateless requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp


LOREM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat duis aute irure dolor in reprehenderit in voluptate velit esse "
    "cillum dolore eu fugiat nulla pariatur"
).split()


@dataclass
class RequestResult:
    kind: str
    session_id: Optional[str]
    rid: str
    turn: int
    status: int
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_device_tokens: int = 0
    cache_host_tokens: int = 0
    cache_storage_tokens: int = 0
    expected_cached_tokens: int = 0
    cache_warning: Optional[str] = None
    error: Optional[str] = None

    @property
    def prefill_proxy_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def observed_cached_tokens(self) -> int:
        return max(
            self.cached_tokens,
            self.cache_device_tokens + self.cache_host_tokens + self.cache_storage_tokens,
        )


def repeated_lorem(target_words: int, prefix: str = "") -> str:
    words = [LOREM_WORDS[i % len(LOREM_WORDS)] for i in range(target_words)]
    return (prefix + "\n" if prefix else "") + " ".join(words)


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[int, dict[str, Any] | str]:
    try:
        async with session.post(url, json=payload, timeout=timeout_s) as resp:
            text = await resp.text()
            try:
                return resp.status, json.loads(text)
            except json.JSONDecodeError:
                return resp.status, text
    except Exception as exc:
        return 0, repr(exc)


async def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(f"{base_url}/health", timeout=5) as resp:
                    if resp.status < 500:
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s")


async def tokenize_count(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    text: str,
) -> Optional[int]:
    status, data = await post_json(
        session,
        f"{base_url}/v1/tokenize",
        {"model": model, "prompt": text, "add_special_tokens": False},
        timeout_s=30,
    )
    if status == 200 and isinstance(data, dict) and isinstance(data.get("count"), int):
        return data["count"]
    return None


async def text_near_token_count(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    target_tokens: int,
    prefix: str,
    use_tokenizer: bool,
) -> str:
    # Word count is a good enough fallback for pressure generation. If the
    # server tokenizer is available, tighten the size with a few probes.
    approx_words = max(1, int(target_tokens * 0.75))
    text = repeated_lorem(approx_words, prefix=prefix)
    if not use_tokenizer:
        return text

    count = await tokenize_count(session, base_url, model, text)
    if not count:
        return text

    for _ in range(4):
        if abs(count - target_tokens) <= max(64, target_tokens // 20):
            break
        scale = target_tokens / max(count, 1)
        approx_words = max(1, int(approx_words * scale))
        text = repeated_lorem(approx_words, prefix=prefix)
        count = await tokenize_count(session, base_url, model, text)
        if not count:
            break
    return text


def build_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "sleep",
                "description": "Wait for a requested number of milliseconds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "milliseconds": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 600000,
                        }
                    },
                    "required": ["milliseconds"],
                },
            },
        }
    ]


def make_user_request(
    rng: random.Random,
    turn: int,
    requested_response: str,
    max_tool_calls: int,
    tool_sleep_ms: int,
    tool_sleep_zero_probability: float,
    tool_sleep_power: float,
) -> tuple[str, int]:
    tool_count = rng.randint(0, max_tool_calls)
    calls = []
    for _ in range(tool_count):
        if rng.random() < tool_sleep_zero_probability:
            milliseconds = 0
        else:
            milliseconds = int((rng.random() ** tool_sleep_power) * tool_sleep_ms)
            milliseconds = max(1, milliseconds)
        calls.append(
            {
                "name": "sleep",
                "arguments": {"milliseconds": milliseconds},
            }
        )
    payload = {
        "turn": turn,
        "requested_tool_calls": calls,
        "final_response": requested_response,
    }
    return (
        "Follow the system instructions exactly for this request:\n"
        + json.dumps(payload, sort_keys=True),
        sum(call["arguments"]["milliseconds"] for call in calls),
    )


def extract_stats(
    kind: str,
    session_id: Optional[str],
    rid: str,
    turn: int,
    status: int,
    latency_s: float,
    data: dict[str, Any] | str,
) -> RequestResult:
    result = RequestResult(
        kind=kind,
        session_id=session_id,
        rid=rid,
        turn=turn,
        status=status,
        latency_s=latency_s,
    )
    if not isinstance(data, dict):
        result.error = str(data)
        return result
    if status >= 400 or "error" in data:
        result.error = json.dumps(data.get("error", data), sort_keys=True)[:1000]
        return result

    usage = data.get("usage") or {}
    result.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    result.completion_tokens = int(usage.get("completion_tokens") or 0)
    result.total_tokens = int(usage.get("total_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    result.cached_tokens = int(details.get("cached_tokens") or 0)

    sglext_details = ((data.get("sglext") or {}).get("cached_tokens_details")) or {}
    result.cache_device_tokens = int(sglext_details.get("device") or 0)
    result.cache_host_tokens = int(sglext_details.get("host") or 0)
    result.cache_storage_tokens = int(sglext_details.get("storage") or 0)
    return result


async def chat_completion(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    messages: list[dict[str, Any]],
    rid: str,
    session_params: Optional[dict[str, Any]],
    kind: str,
    turn: int,
    session_id: Optional[str],
) -> RequestResult:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "rid": rid,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "ignore_eos": True,
        "return_cached_tokens_details": True,
    }
    if session_params is not None:
        payload["session_params"] = session_params
    if args.include_tools:
        payload["tools"] = build_tools()
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = True

    async with semaphore:
        start = time.perf_counter()
        status, data = await post_json(
            session,
            f"{args.base_url}/v1/chat/completions",
            payload,
            timeout_s=args.request_timeout_s,
        )
        latency_s = time.perf_counter() - start
    return extract_stats(kind, session_id, rid, turn, status, latency_s, data)


async def open_session(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    session_id: str,
    capacity: int,
) -> None:
    status, data = await post_json(
        session,
        f"{args.base_url}/open_session",
        {
            "session_id": session_id,
            "capacity_of_str_len": capacity,
            "timeout": args.session_timeout_s,
        },
        timeout_s=args.request_timeout_s,
    )
    if status >= 400 or status == 0:
        raise RuntimeError(f"open_session failed for {session_id}: {data}")


async def close_session(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    session_id: str,
) -> None:
    await post_json(
        session,
        f"{args.base_url}/close_session",
        {"session_id": session_id},
        timeout_s=args.request_timeout_s,
    )


async def run_session_trajectory(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    session_idx: int,
    header: str,
    session_seed_text: str,
) -> list[RequestResult]:
    rng = random.Random(args.seed + session_idx * 9973)
    session_id = f"session-cache-pressure-{args.run_id}-{session_idx:05d}"
    await open_session(
        session,
        args,
        session_id,
        capacity=max(8192, args.session_capacity_chars),
    )

    results: list[RequestResult] = []
    previous_rid: Optional[str] = None
    previous_result: Optional[RequestResult] = None
    try:
        for turn in range(args.turns):
            rid = f"{session_id}-turn-{turn:04d}"
            response_text = f"session {session_idx} turn {turn} complete"
            user_text, sleep_ms = make_user_request(
                rng,
                turn,
                response_text,
                args.max_tool_calls,
                args.tool_sleep_ms,
                args.tool_sleep_zero_probability,
                args.tool_sleep_power,
            )
            turn_pad = repeated_lorem(
                max(1, int(args.turn_prompt_tokens * 0.75)),
                prefix=f"TURN GUID {uuid.uuid5(uuid.NAMESPACE_DNS, rid)}",
            )
            user_text = turn_pad + "\n\n" + user_text
            if turn == 0:
                messages = [
                    {"role": "system", "content": header},
                    {
                        "role": "user",
                        "content": session_seed_text + "\n\n" + user_text,
                    },
                ]
                session_params = {"id": session_id}
            else:
                messages = [{"role": "user", "content": user_text}]
                session_params = {"id": session_id, "rid": previous_rid}

            result = await chat_completion(
                session,
                args,
                semaphore,
                messages,
                rid,
                session_params,
                kind="session",
                turn=turn,
                session_id=session_id,
            )
            if (
                previous_result is not None
                and previous_result.error is None
                and 200 <= previous_result.status < 300
                and result.error is None
                and 200 <= result.status < 300
            ):
                result.expected_cached_tokens = previous_result.total_tokens
                if result.observed_cached_tokens < result.expected_cached_tokens:
                    result.cache_warning = (
                        f"expected at least {result.expected_cached_tokens} cached "
                        f"tokens from previous session turn, got "
                        f"{result.observed_cached_tokens}"
                    )
                    print(
                        "cache warning: "
                        f"session={session_id} turn={turn} rid={rid} "
                        f"prev_rid={previous_result.rid} "
                        f"expected_cached>={result.expected_cached_tokens} "
                        f"observed_cached={result.observed_cached_tokens}",
                        flush=True,
                    )
            if args.verbose_requests:
                print(
                    f"session turn: session={session_id} turn={turn} rid={rid} "
                    f"prev_rid={previous_rid} status={result.status} "
                    f"prompt={result.prompt_tokens} completion={result.completion_tokens} "
                    f"cached={result.cached_tokens} "
                    f"cached_details={result.cache_device_tokens}/"
                    f"{result.cache_host_tokens}/{result.cache_storage_tokens}",
                    flush=True,
                )
            results.append(result)
            previous_result = result
            previous_rid = rid
            if args.simulate_tool_sleeps and sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000.0)
    finally:
        if args.close_sessions:
            await close_session(session, args, session_id)
    return results


async def run_stateless_request(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    request_idx: int,
    header: str,
) -> RequestResult:
    rng = random.Random(args.seed + 1000003 + request_idx)
    rid = f"session-cache-pressure-{args.run_id}-stateless-{request_idx:05d}"
    response_text = f"stateless {request_idx} complete"
    user_text, sleep_ms = make_user_request(
        rng,
        request_idx,
        response_text,
        args.max_tool_calls,
        args.tool_sleep_ms,
        args.tool_sleep_zero_probability,
        args.tool_sleep_power,
    )
    prompt_pad = repeated_lorem(
        max(1, int(args.stateless_prompt_tokens * 0.75)),
        prefix=f"STATELESS GUID {uuid.uuid5(uuid.NAMESPACE_DNS, rid)}",
    )
    messages = [
        {"role": "system", "content": header},
        {"role": "user", "content": prompt_pad + "\n\n" + user_text},
    ]
    result = await chat_completion(
        session,
        args,
        semaphore,
        messages,
        rid,
        session_params=None,
        kind="stateless",
        turn=request_idx,
        session_id=None,
    )
    if args.simulate_tool_sleeps and sleep_ms > 0:
        await asyncio.sleep(sleep_ms / 1000.0)
    return result


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil((p / 100.0) * len(values)) - 1))
    return values[idx]


def summarize_flat(results: list[RequestResult], warning_count: int = 0) -> dict[str, Any]:
    ok = [r for r in results if r.error is None and 200 <= r.status < 300]
    cache_warnings = [r for r in results if r.cache_warning]
    latencies = [r.latency_s for r in ok]
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    cached_tokens = sum(r.cached_tokens for r in ok)
    prefill_proxy_tokens = sum(r.prefill_proxy_tokens for r in ok)
    device_tokens = sum(r.cache_device_tokens for r in ok)
    host_tokens = sum(r.cache_host_tokens for r in ok)
    storage_tokens = sum(r.cache_storage_tokens for r in ok)

    flat = {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "latency_avg_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit_ratio": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        "prefill_proxy_tokens": prefill_proxy_tokens,
        "cache_device_tokens": device_tokens,
        "cache_host_tokens": host_tokens,
        "cache_storage_tokens": storage_tokens,
        "cache_warnings": len(cache_warnings),
    }
    if warning_count:
        flat["latest_eviction_warnings"] = warning_count

    return flat


def summarize_results(
    results: list[RequestResult],
    warning_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({r.kind for r in results}):
        subset = [r for r in results if r.kind == kind]
        by_kind[kind] = summarize_flat(subset)

    flat = summarize_flat(results, warning_count)
    return {
        "_flat": flat,
        "run_id": args.run_id,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"server_arg"}
        },
        "summary": flat,
        "by_kind": by_kind,
        "latest_eviction_warnings": warning_count,
    }


def scan_warning_count(log_path: Optional[Path], pattern: str) -> int:
    if log_path is None or not log_path.exists():
        return 0
    regex = re.compile(pattern)
    count = 0
    with log_path.open("r", errors="replace") as f:
        for line in f:
            if regex.search(line):
                count += 1
    return count


def start_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    if not args.model_path:
        raise ValueError("--model-path is required with --start-server")
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    cmd.extend(args.server_arg or [])
    if "--enable-cache-report" not in cmd:
        cmd.append("--enable-cache-report")

    env = os.environ.copy()
    if args.old_lru_mode:
        env["SGLANG_HIMAMBA_DISABLE_SESSION_RETENTION_PRIORITY"] = "1"
    if args.warn_on_latest_evict:
        env["SGLANG_HIMAMBA_WARN_ON_LATEST_RETENTION_EVICT"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )


async def build_headers_and_session_texts(
    args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    async with aiohttp.ClientSession() as session:
        headers = []
        for i in range(args.common_headers):
            guid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{args.run_id}-header-{i}")
            prefix = (
                f"SYSTEM PROMPT GUID: {guid}\n"
                "For each user request, make exactly the requested tool calls if "
                "the chat template supports tools. Then reply with the given "
                "final_response text and no extra prose."
            )
            headers.append(
                await text_near_token_count(
                    session,
                    args.base_url,
                    args.model,
                    args.common_header_tokens,
                    prefix,
                    args.use_tokenizer_sizing,
                )
            )

        session_texts = []
        for i in range(args.sessions):
            guid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{args.run_id}-session-{i}")
            prefix = f"SESSION FACTS GUID: {guid}"
            session_texts.append(
                await text_near_token_count(
                    session,
                    args.base_url,
                    args.model,
                    args.session_seed_tokens,
                    prefix,
                    args.use_tokenizer_sizing,
                )
            )
    return headers, session_texts


def estimate_pressure(args: argparse.Namespace) -> dict[str, Any]:
    first_turn = args.common_header_tokens + args.session_seed_tokens
    later_turn = args.turn_prompt_tokens + args.max_tokens
    session_tokens = args.sessions * (first_turn + max(0, args.turns - 1) * later_turn)
    stateless_tokens = args.stateless_requests * (
        args.common_header_tokens + args.stateless_prompt_tokens
    )
    total = session_tokens + stateless_tokens

    suggested_turns = args.turns
    if args.target_pressure_tokens:
        per_round = max(1, args.sessions * later_turn)
        missing = max(0, args.target_pressure_tokens - args.sessions * first_turn)
        suggested_turns = max(args.turns, 1 + math.ceil(missing / per_round))

    return {
        "estimated_prompt_pressure_tokens": total,
        "estimated_session_tokens": session_tokens,
        "estimated_stateless_tokens": stateless_tokens,
        "suggested_turns_for_target": suggested_turns,
    }


async def run_workload(args: argparse.Namespace) -> list[RequestResult]:
    pressure = estimate_pressure(args)
    print(json.dumps({"pressure_estimate": pressure}, indent=2), flush=True)
    if args.auto_turns and pressure["suggested_turns_for_target"] > args.turns:
        args.turns = pressure["suggested_turns_for_target"]
        print(f"auto-adjusted turns to {args.turns}", flush=True)

    headers, session_texts = await build_headers_and_session_texts(args)
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 16))
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(args.sessions):
            tasks.append(
                asyncio.create_task(
                    run_session_trajectory(
                        session,
                        args,
                        semaphore,
                        i,
                        headers[i % len(headers)],
                        session_texts[i],
                    )
                )
            )
        for i in range(args.stateless_requests):
            tasks.append(
                asyncio.create_task(
                    run_stateless_request(
                        session,
                        args,
                        semaphore,
                        i,
                        headers[i % len(headers)],
                    )
                )
            )

        results: list[RequestResult] = []
        completed = 0
        for task in asyncio.as_completed(tasks):
            item = await task
            if isinstance(item, list):
                results.extend(item)
            else:
                results.append(item)
            completed += 1
            if completed % max(1, args.progress_every) == 0:
                print(f"completed task groups: {completed}/{len(tasks)}", flush=True)
        return results


def write_outputs(
    args: argparse.Namespace,
    results: list[RequestResult],
    warning_count: int,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / f"{args.run_id}.jsonl"
    with details_path.open("w") as f:
        for result in results:
            f.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    summary = summarize_results(results, warning_count, args)
    summary.pop("_flat", None)
    summary_path = args.output_dir / f"{args.run_id}.summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(summary["summary"], indent=2, sort_keys=True), flush=True)
    print(f"wrote details: {details_path}", flush=True)
    print(f"wrote summary: {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic session cache pressure workload for SGLang."
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="default")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--ready-timeout-s", type=float, default=600)
    parser.add_argument("--request-timeout-s", type=float, default=600)
    parser.add_argument("--old-lru-mode", action="store_true")
    parser.add_argument("--warn-on-latest-evict", action="store_true")
    parser.add_argument("--verbose-requests", action="store_true")
    parser.add_argument(
        "--eviction-warning-regex",
        default=r"latest.*evict|evict.*latest|session_latest.*evict",
    )
    parser.add_argument("--server-log-file", type=Path, default=None)

    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--common-headers", type=int, default=4)
    parser.add_argument("--common-header-tokens", type=int, default=2048)
    parser.add_argument("--session-seed-tokens", type=int, default=1024)
    parser.add_argument("--turn-prompt-tokens", type=int, default=256)
    parser.add_argument("--stateless-requests", type=int, default=64)
    parser.add_argument("--stateless-prompt-tokens", type=int, default=4096)
    parser.add_argument("--target-pressure-tokens", type=int, default=0)
    parser.add_argument("--auto-turns", action="store_true")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--tool-sleep-ms", type=int, default=2000)
    parser.add_argument("--tool-sleep-zero-probability", type=float, default=0.0)
    parser.add_argument("--tool-sleep-power", type=float, default=1.0)
    parser.add_argument("--simulate-tool-sleeps", action="store_true", default=True)
    parser.add_argument(
        "--no-simulate-tool-sleeps",
        action="store_false",
        dest="simulate_tool_sleeps",
    )
    parser.add_argument("--include-tools", action="store_true")
    parser.add_argument("--session-capacity-chars", type=int, default=4_000_000)
    parser.add_argument("--session-timeout-s", type=float, default=3600)
    parser.add_argument("--close-sessions", action="store_true")
    parser.add_argument("--use-tokenizer-sizing", action="store_true", default=True)
    parser.add_argument(
        "--no-use-tokenizer-sizing",
        action="store_false",
        dest="use_tokenizer_sizing",
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mamba-radix-cache/results"),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = time.strftime("session-cache-pressure-%Y%m%d-%H%M%S")
    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"
    return args


async def async_main() -> None:
    args = parse_args()
    server_proc: Optional[subprocess.Popen] = None
    launched_log_path: Optional[Path] = None
    try:
        if args.start_server:
            launched_log_path = args.output_dir / f"{args.run_id}.server.log"
            server_proc = start_server(args, launched_log_path)
            args.server_log_file = launched_log_path
        await wait_for_server(args.base_url, args.ready_timeout_s)
        results = await run_workload(args)
        warning_count = scan_warning_count(
            args.server_log_file,
            args.eviction_warning_regex,
        )
        write_outputs(args, results, warning_count)
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=30)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
