import bisect
import re
from typing import Dict, List, Optional, Tuple

import pdfplumber


def load_pdf_pages(path: str) -> List[Dict]:
    """Load PDF and return a list of pages with text and metadata.

    Returns a list of dicts: {"page": int, "text": str}
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
    return pages


MIN_PAGE_TEXT_CHARS = 20


def flagged_low_text_pages(pages: List[Dict], min_chars: int = MIN_PAGE_TEXT_CHARS) -> List[int]:
    """Page numbers whose extracted text is empty or suspiciously short.

    Likely scanned/image-only pages, or a layout pdfplumber couldn't parse.
    """
    return [p["page"] for p in pages if len(p.get("text", "").strip()) < min_chars]


_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # naive sentence split by punctuation + whitespace
    sentences = _SENTENCE_SPLIT.split(text)
    # strip and filter
    return [s.strip() for s in sentences if s.strip()]


def format_page_label(source: dict) -> Optional[str]:
    """Render a chunk's page(s) as a display label: "3", or "3-4" when it spans pages."""
    start_page = source.get('start_page')
    if start_page is None:
        return None
    end_page = source.get('end_page')
    if end_page is not None and end_page != start_page:
        return f"{start_page}-{end_page}"
    return str(start_page)


def _build_document(pages: List[Dict]) -> Tuple[str, List[Tuple[int, int]]]:
    """Concatenate page texts into one document string, plus an offset->page lookup.

    Returns (document_text, page_offsets), where page_offsets is a list of
    (start_offset, page_num) pairs -- one per non-empty page, sorted by offset.
    """
    parts = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for page in pages:
        text = page.get("text", "")
        if not text:
            continue
        offsets.append((cursor, page.get("page")))
        parts.append(text)
        cursor += len(text) + 1  # +1 for the "\n" join separator
    return "\n".join(parts), offsets


def _page_at(offset: int, page_offsets: List[Tuple[int, int]]) -> int:
    """Binary-search the page owning a character offset in the concatenated document."""
    starts = [o for o, _ in page_offsets]
    idx = max(bisect.bisect_right(starts, offset) - 1, 0)
    return page_offsets[idx][1]


def chunk_pages_to_chunks(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Create overlapping chunks across the whole document, preserving page attribution.

    Chunks are cut on chunk_size/overlap alone -- page boundaries no longer force a
    cut, so a sentence that spans two pages stays in one chunk. Each chunk is a dict:
    {"text": str, "source": {"start_page": int, "end_page": int, "start_char": int, "end_char": int}}.
    start_char/end_char index into the concatenated document, not a single page's text.
    """
    document, page_offsets = _build_document(pages)
    if not document:
        return []

    sentences = split_sentences(document)
    chunks: List[Dict] = []
    cur = ""
    cur_start = 0
    pos = 0

    for sent in sentences:
        sent_start = pos if not cur else pos + 1
        if cur:
            next_text = cur + " " + sent
        else:
            next_text = sent
            cur_start = sent_start

        if len(next_text) >= chunk_size:
            start_idx = cur_start
            end_idx = start_idx + len(cur)
            chunks.append(
                {
                    "text": cur.strip(),
                    "source": {
                        "start_page": _page_at(start_idx, page_offsets),
                        "end_page": _page_at(max(end_idx - 1, start_idx), page_offsets),
                        "start_char": start_idx,
                        "end_char": end_idx,
                    },
                }
            )
            # prepare next chunk with overlap
            overlap_text = next_text[-overlap:]
            cur = overlap_text
            sent_end = sent_start + len(sent)
            cur_start = sent_end - len(overlap_text)
            pos = sent_end
        else:
            cur = next_text
            pos = sent_start + len(sent)

    if cur:
        end_idx = cur_start + len(cur)
        chunks.append(
            {
                "text": cur.strip(),
                "source": {
                    "start_page": _page_at(cur_start, page_offsets),
                    "end_page": _page_at(max(end_idx - 1, cur_start), page_offsets),
                    "start_char": cur_start,
                    "end_char": end_idx,
                },
            }
        )

    return chunks


def load_and_chunk(path: str, chunk_size: int = 1000, overlap: int = 200) -> Tuple[List[Dict], List[int]]:
    """Load a PDF, chunk it, and report which pages produced little or no text.

    Returns (chunks, flagged_pages) -- flagged_pages are page numbers likely to be
    scanned images or layouts pdfplumber couldn't parse (see flagged_low_text_pages).
    """
    pages = load_pdf_pages(path)
    chunks = chunk_pages_to_chunks(pages, chunk_size=chunk_size, overlap=overlap)
    return chunks, flagged_low_text_pages(pages)
