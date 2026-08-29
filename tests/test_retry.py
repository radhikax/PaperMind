from src.retry import call_with_retries_validate


def test_returns_immediately_when_no_validator():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result = call_with_retries_validate(fn, max_attempts=3, initial_delay=0)
    assert result == "ok"
    assert len(calls) == 1


def test_retries_until_validator_passes():
    calls = []

    def fn():
        calls.append(1)
        return len(calls)

    result = call_with_retries_validate(fn, validator=lambda r: r >= 3, max_attempts=5, initial_delay=0)
    assert result == 3
    assert len(calls) == 3


def test_gives_up_after_max_attempts_and_returns_last_result():
    calls = []

    def fn():
        calls.append(1)
        return "bad"

    result = call_with_retries_validate(fn, validator=lambda r: r == "good", max_attempts=2, initial_delay=0)
    assert result == "bad"
    assert len(calls) == 2


def test_on_attempt_callback_invoked_per_try():
    seen = []

    def fn():
        return "bad"

    call_with_retries_validate(
        fn,
        validator=lambda r: r == "good",
        max_attempts=3,
        initial_delay=0,
        on_attempt=lambda a: seen.append(a),
    )
    assert seen == [1, 2, 3]


def test_reraises_after_exhausting_attempts_on_exception():
    calls = []

    def fn():
        calls.append(1)
        raise RuntimeError("boom")

    try:
        call_with_retries_validate(fn, max_attempts=2, initial_delay=0)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert len(calls) == 2
