import tempfile
from pathlib import Path

from reportlab.pdfgen import canvas

from src.ingest import load_and_chunk


def make_sample_pdf(path: str, text: str = "Hello world. This is a test PDF."):
    c = canvas.Canvas(path)
    c.drawString(72, 720, text)
    c.save()


def test_load_and_chunk():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    make_sample_pdf(tmp.name, "This is a simple PDF for testing ingestion. It has a few sentences.")
    chunks = load_and_chunk(tmp.name, chunk_size=200, overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for c in chunks:
        assert "text" in c and "source" in c
