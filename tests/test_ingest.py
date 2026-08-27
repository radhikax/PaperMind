import tempfile

from reportlab.pdfgen import canvas

from src.ingest import (flagged_low_text_pages, format_page_label,
                        load_and_chunk)


def make_sample_pdf(path: str, text: str = "Hello world. This is a test PDF."):
    c = canvas.Canvas(path)
    c.drawString(72, 720, text)
    c.save()


def test_flagged_low_text_pages_flags_empty_and_near_empty():
    pages = [
        {"page": 1, "text": "Plenty of real content on this page to pass the threshold easily."},
        {"page": 2, "text": ""},
        {"page": 3, "text": "x"},
    ]
    assert flagged_low_text_pages(pages) == [2, 3]


def test_flagged_low_text_pages_empty_when_all_pages_have_content():
    pages = [{"page": 1, "text": "This page has more than twenty characters of real text."}]
    assert flagged_low_text_pages(pages) == []


def test_format_page_label_single_page():
    assert format_page_label({"start_page": 3, "end_page": 3}) == "3"


def test_format_page_label_spanning_pages():
    assert format_page_label({"start_page": 3, "end_page": 4}) == "3-4"


def test_format_page_label_missing_start_page_is_none():
    assert format_page_label({}) is None


def test_load_and_chunk():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    make_sample_pdf(tmp.name, "This is a simple PDF for testing ingestion. It has a few sentences.")
    chunks = load_and_chunk(tmp.name, chunk_size=200, overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for c in chunks:
        assert "text" in c and "source" in c
