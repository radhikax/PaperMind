from scripts.run_eval import GOLDEN_QA_PATH, load_golden_qa


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
