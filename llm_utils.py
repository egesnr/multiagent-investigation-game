"""
Shared retry wrapper for LLM calls.

The underlying SDK already retries transient errors internally (a handful of
fast retries within a couple seconds), but a real provider-side capacity
outage on the free tier can outlast that whole window. With ~9 calls per
turn, even a moderate per-call failure rate compounds into a turn crashing
partway through — losing an already-scored turn's state, since run_turn only
returns once the whole turn (scoring through speaker) completes. This adds a
slower, longer-horizon retry on top so a multi-second blip doesn't take the
whole turn down with it.
"""

import time

# Errors where retrying literally cannot succeed, so waiting and trying again
# is pure wasted time: a model name that doesn't exist won't start existing,
# and a per-day quota that's already exhausted won't refill inside this
# process's lifetime. Only rate limits / transient server errors are worth
# the backoff above.
_NON_RETRYABLE_MARKERS = ("NotFound", "PerDay")


def _is_retryable(exc: Exception) -> bool:
    exc_text = f"{exc.__class__.__name__} {exc}"
    return not any(marker in exc_text for marker in _NON_RETRYABLE_MARKERS)


# 2/4/8/16 rather than 8/16/32/64. Measured on a bad free-tier spell: three
# calls in one turn hit the retry path, and the old schedule spent 56 seconds
# waiting on each of them — 704 seconds for a turn whose actual work was under
# 40. These errors clear in a second or two when they clear at all, so the long
# tail of the old schedule bought nothing and cost most of the turn.
# (label, seconds, retries, seconds spent waiting between retries) for every
# call since the last reset — run_turn prints it, so a slow turn says whether
# the time went to the model, to rate-limit retries, or to our own code.
STEP_TIMES: list[tuple[str, float, int, float]] = []


def invoke_with_retry(chain, payload: dict, attempts: int = 5, base_delay: float = 2.0, label: str = "llm"):
    last_exc = None
    started = time.perf_counter()
    waited = 0.0
    for attempt in range(attempts):
        try:
            result = chain.invoke(payload)
            STEP_TIMES.append((label, time.perf_counter() - started, attempt, waited))
            return result
        except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised if retries exhaust
            last_exc = exc
            if attempt == attempts - 1 or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[retry] LLM call failed ({exc.__class__.__name__}), "
                  f"retrying in {delay:.0f}s (attempt {attempt + 2}/{attempts})...")
            time.sleep(delay)
            waited += delay
    raise last_exc
