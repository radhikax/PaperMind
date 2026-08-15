import re
from typing import Dict, List

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


_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # naive sentence split by punctuation + whitespace
    sentences = _SENTENCE_SPLIT.split(text)
    # strip and filter
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages_to_chunks(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Create overlapping chunks across pages, preserving source metadata.

    Each chunk is a dict: {"text": str, "source": {"page": int, "start_char": int, "end_char": int}}
    """
    chunks: List[Dict] = []
    for page in pages:
        page_num = page.get("page")
        text = page.get("text", "")
        if not text:
            continue
        sentences = split_sentences(text)
        cur = ""
        for sent in sentences:
            if cur:
                next_text = cur + " " + sent
            else:
                next_text = sent

            if len(next_text) >= chunk_size:
                # finalize current chunk
                start_idx = text.find(cur) if cur else text.find(sent)
                end_idx = start_idx + len(cur)
                chunks.append(
                    {
                        "text": cur.strip(),
                        "source": {
                            "page": page_num,
                            "start_char": start_idx,
                            "end_char": end_idx,
                        },
                    }
                )
                # prepare next chunk with overlap
                overlap_text = next_text[-overlap:]
                cur = overlap_text
            else:
                cur = next_text

        if cur:
            start_idx = text.find(cur)
            end_idx = start_idx + len(cur)
            chunks.append(
                {
                    "text": cur.strip(),
                    "source": {
                        "page": page_num,
                        "start_char": start_idx,
                        "end_char": end_idx,
                    },
                }
            )

    return chunks


def load_and_chunk(path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    pages = load_pdf_pages(path)
    return chunk_pages_to_chunks(pages, chunk_size=chunk_size, overlap=overlap)
