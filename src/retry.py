import time
from typing import Any, Callable, Optional


def call_with_retries_validate(
    fn: Callable[..., Any],
    *args: Any,
    validator: Optional[Callable[[Any], bool]] = None,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    on_attempt: Optional[Callable[[int], None]] = None,
    **kwargs: Any,
) -> Any:
    """Call fn(*args, **kwargs) repeatedly until validator(result) is True or attempts exhausted."""
    delay = initial_delay
    result = None
    for attempt in range(1, max_attempts + 1):
        if on_attempt:
            try:
                on_attempt(attempt)
            except Exception:
                pass
        try:
            result = fn(*args, **kwargs)
        except Exception:
            if attempt < max_attempts:
                time.sleep(delay)
                delay *= multiplier
                continue
            raise

        if validator is None:
            return result

        try:
            ok = validator(result)
        except Exception:
            ok = False

        if ok:
            return result
        if attempt < max_attempts:
            time.sleep(delay)
            delay *= multiplier
            continue
    return result
