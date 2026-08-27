"""Resolves a paper_id to a stable on-disk index path, and remembers which
papers have been indexed so the UI can offer them back without re-uploading.

One index per paper (see docs/superpowers/plans/2026-08-23-improvement-roadmap.md,
Phase 1) instead of the single global paper_index.index/.pkl this replaces.
"""
import hashlib
import os

DEFAULT_BASE_DIR = "indexes"


def slugify_paper_id(paper_id: str) -> str:
    return hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:16]


def index_path_for(paper_id: str, base_dir: str = DEFAULT_BASE_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, slugify_paper_id(paper_id))
