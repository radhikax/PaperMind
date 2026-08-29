from scripts.run_eval import (GOLDEN_QA_PATH, MIN_VERIFIED_RATIO,
                              chunk_covers_expected_page, load_golden_qa,
                              passes_verification_floor)


def test_load_golden_qa_returns_all_twelve_questions():
    questions = load_golden_qa(GOLDEN_QA_PATH)
    assert len(questions) == 12


def test_load_golden_qa_entries_have_required_keys():
    questions = load_golden_qa(GOLDEN_QA_PATH)
    for q in questions:
        assert "question" in q
        assert "expected_pages" in q
        assert "adversarial" in q


def test_load_golden_qa_has_exactly_one_adversarial_question():
    questions = load_golden_qa(GOLDEN_QA_PATH)
    adversarial = [q for q in questions if q["adversarial"]]
    assert len(adversarial) == 1
    assert adversarial[0]["expected_pages"] == []


def test_chunk_covers_expected_page_true_when_page_in_range():
    assert chunk_covers_expected_page({"start_page": 2, "end_page": 4}, [3]) is True


def test_chunk_covers_expected_page_false_when_page_outside_range():
    assert chunk_covers_expected_page({"start_page": 2, "end_page": 4}, [5]) is False


def test_chunk_covers_expected_page_false_when_no_expected_pages():
    assert chunk_covers_expected_page({"start_page": 2, "end_page": 4}, []) is False


def test_chunk_covers_expected_page_false_when_source_missing_pages():
    assert chunk_covers_expected_page({}, [1]) is False


def test_passes_verification_floor_at_or_above_threshold():
    assert passes_verification_floor(MIN_VERIFIED_RATIO) is True
    assert passes_verification_floor(1.0) is True


def test_passes_verification_floor_below_threshold():
    assert passes_verification_floor(0.1) is False
