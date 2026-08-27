import tempfile

import pdfplumber

from scripts.fixtures.golden_paper import PAGES, build_golden_paper


def test_pages_constant_has_five_entries():
    assert len(PAGES) == 5


def test_build_golden_paper_produces_five_pages():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    build_golden_paper(tmp.name)
    with pdfplumber.open(tmp.name) as pdf:
        assert len(pdf.pages) == 5


def test_build_golden_paper_page_content_matches_known_facts():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    build_golden_paper(tmp.name)
    with pdfplumber.open(tmp.name) as pdf:
        texts = [p.extract_text() or "" for p in pdf.pages]
    assert "94.2%" in texts[0]
    assert "12,400" in texts[1]
    assert "0.0003" in texts[2]
    assert "86.1%" in texts[3]
    assert "controlled lighting" in texts[4]
