"""Resolves a paper_id to a stable on-disk index path, and remembers which
papers have been indexed so the UI can offer them back without re-uploading.

One index per paper (see docs/superpowers/plans/2026-08-23-improvement-roadmap.md,
Phase 1) instead of the single global paper_index.index/.pkl this replaces.
"""
import hashlib
import json
import os
from typing import List

DEFAULT_BASE_DIR = "indexes"
MANIFEST_FILENAME = "manifest.json"


def slugify_paper_id(paper_id: str) -> str:
    return hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:16]


def index_path_for(paper_id: str, base_dir: str = DEFAULT_BASE_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, slugify_paper_id(paper_id))


def _manifest_path(base_dir: str) -> str:
    return os.path.join(base_dir, MANIFEST_FILENAME)


def _load_manifest(base_dir: str) -> dict:
    path = _manifest_path(base_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def register_paper(paper_id: str, display_name: str, base_dir: str = DEFAULT_BASE_DIR) -> None:
    os.makedirs(base_dir, exist_ok=True)
    manifest = _load_manifest(base_dir)
    manifest[slugify_paper_id(paper_id)] = {"paper_id": paper_id, "display_name": display_name}
    with open(_manifest_path(base_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def list_registered_papers(base_dir: str = DEFAULT_BASE_DIR) -> List[dict]:
    manifest = _load_manifest(base_dir)
    return [
        {"slug": slug, "paper_id": entry["paper_id"], "display_name": entry["display_name"]}
        for slug, entry in manifest.items()
    ]
