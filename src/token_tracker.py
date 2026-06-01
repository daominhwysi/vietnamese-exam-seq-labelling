"""
token_tracker.py — Logs DeepSeek API token usage to logs/.

Each API call appends one JSON line to:
  logs/token_usage_<YYYY-MM-DD>.jsonl

Each record contains:
  - timestamp         ISO-8601 UTC timestamp
  - model             Model name used
  - input_tokens      Tokens in the prompt (input)
  - reasoning_tokens  Chain-of-thought tokens (thinking), billed as output
  - output_tokens     Actual response content tokens (excludes reasoning)

Note: the API's raw completion_tokens = reasoning_tokens + output_tokens.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── thread-safe write lock ────────────────────────────────────────────────────
_write_lock = threading.Lock()

# ── log directory (project-root/logs/) ───────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def _get_log_path() -> Path:
    """Return today's JSONL log file path, creating the directory if needed."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return _LOGS_DIR / f"token_usage_{date_str}.jsonl"


def log_response(response: Any, model: str | None = None) -> None:
    """
    Append a token-usage record for *response* to today's JSONL log.

    Parameters
    ----------
    response:
        The ``ChatCompletion`` object returned by
        ``client.chat.completions.create()``.
    model:
        Override the model name. If *None*, taken from ``response.model``.
    """
    usage = getattr(response, "usage", None)

    # Reasoning tokens live inside completion_tokens_details (DeepSeek-specific)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens: int = getattr(details, "reasoning_tokens", 0) or 0

    completion_tokens: int = getattr(usage, "completion_tokens", 0) or 0
    # output_tokens = only the visible response, not the chain-of-thought
    output_tokens: int = completion_tokens - reasoning_tokens

    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model": model or getattr(response, "model", "unknown"),
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
    }

    log_path = _get_log_path()
    with _write_lock:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── summary helpers ───────────────────────────────────────────────────────────

def load_logs(date: str | None = None) -> list[dict]:
    """
    Load all records from a log file.

    Parameters
    ----------
    date:
        ``"YYYY-MM-DD"`` string. Defaults to today (UTC).
    """
    if date is None:
        date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    log_path = _LOGS_DIR / f"token_usage_{date}.jsonl"
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def summarize(date: str | None = None) -> dict:
    """
    Return aggregate token statistics for all calls logged on *date*.

    Returns a dict with keys: calls, input_tokens, reasoning_tokens, output_tokens.
    """
    records = load_logs(date)
    summary: dict[str, int] = {
        "calls": len(records),
        "input_tokens": 0,
        "reasoning_tokens": 0,
        "output_tokens": 0,
    }
    for r in records:
        summary["input_tokens"] += r.get("input_tokens", 0)
        summary["reasoning_tokens"] += r.get("reasoning_tokens", 0)
        summary["output_tokens"] += r.get("output_tokens", 0)
    return summary


def print_summary(date: str | None = None) -> None:
    """Print a human-readable token usage summary for *date* (default: today)."""
    if date is None:
        date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    s = summarize(date)
    print(f"\n── Token Usage Summary ({date}) ──────────────")
    print(f"  API calls        : {s['calls']}")
    print(f"  Input tokens     : {s['input_tokens']:,}")
    print(f"  Reasoning tokens : {s['reasoning_tokens']:,}")
    print(f"  Output tokens    : {s['output_tokens']:,}")
    print("────────────────────────────────────────────\n")


if __name__ == "__main__":
    print_summary()
