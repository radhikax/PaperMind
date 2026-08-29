import json

from src.calibration import (fit_calibrator, isotonic_regression,
                             load_calibrator, load_feedback)


def _write_feedback(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_isotonic_regression_leaves_nondecreasing_input_unchanged():
    assert isotonic_regression([0.1, 0.5, 0.9], [1, 1, 1]) == [0.1, 0.5, 0.9]


def test_isotonic_regression_pools_a_violating_pair():
    fitted = isotonic_regression([0.8, 0.2], [1, 1])
    assert fitted[0] == fitted[1] == 0.5


def test_load_feedback_skips_malformed_and_unlabeled_lines(tmp_path):
    path = tmp_path / "feedback.jsonl"
    path.write_text(
        '{"confidence": 70, "label": "accurate"}\n'
        "not json\n"
        '{"confidence": 40, "label": "unknown"}\n'
        '{"label": "accurate"}\n',
        encoding="utf-8",
    )
    records = load_feedback(str(path))
    assert len(records) == 1
    assert records[0]["confidence"] == 70


def test_calibrator_inactive_below_min_samples_passes_raw_score_through():
    records = [{"confidence": 90, "label": "hallucinated"}] * 3
    calibrator = fit_calibrator(records, min_samples=8)
    assert calibrator.active is False
    assert calibrator.apply(90) == 90


def test_calibrator_active_corrects_an_overconfident_bucket():
    # Historically, scores in the 80-100 bucket turned out accurate only 20% of the time.
    records = (
        [{"confidence": 90, "label": "hallucinated"}] * 8
        + [{"confidence": 90, "label": "accurate"}] * 2
    )
    calibrator = fit_calibrator(records, min_samples=8)
    assert calibrator.active is True
    calibrated = calibrator.apply(90)
    assert calibrated < 90
    assert 15 <= calibrated <= 25


def test_calibrator_is_monotonic_across_buckets():
    records = (
        [{"confidence": 10, "label": "accurate"}] * 9
        + [{"confidence": 90, "label": "hallucinated"}] * 9
    )
    calibrator = fit_calibrator(records, min_samples=8)
    assert calibrator.apply(10) >= calibrator.apply(90)


def test_load_calibrator_reads_from_disk(tmp_path):
    path = tmp_path / "feedback.jsonl"
    _write_feedback(
        path,
        [{"confidence": 50, "label": "accurate"}] * 10,
    )
    calibrator = load_calibrator(str(path))
    assert calibrator.n_samples == 10
    assert calibrator.active is True


def test_load_calibrator_missing_file_is_inactive(tmp_path):
    calibrator = load_calibrator(str(tmp_path / "does-not-exist.jsonl"))
    assert calibrator.n_samples == 0
    assert calibrator.active is False
    assert calibrator.apply(55) == 55
