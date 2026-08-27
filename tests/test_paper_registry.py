import os

from src.paper_registry import index_path_for, slugify_paper_id


def test_slugify_is_stable_for_the_same_paper_id():
    assert slugify_paper_id("paper:foo.pdf") == slugify_paper_id("paper:foo.pdf")


def test_slugify_differs_for_different_paper_ids():
    assert slugify_paper_id("paper:foo.pdf") != slugify_paper_id("paper:bar.pdf")


def test_slugify_is_filesystem_safe():
    slug = slugify_paper_id("paper:weird name / with * chars?.pdf")
    assert all(c.isalnum() for c in slug)


def test_index_path_for_creates_base_dir(tmp_path):
    base_dir = str(tmp_path / "indexes")
    assert not os.path.exists(base_dir)
    path = index_path_for("paper:foo.pdf", base_dir=base_dir)
    assert os.path.isdir(base_dir)
    assert path.startswith(base_dir)


def test_index_path_for_is_stable_across_calls(tmp_path):
    base_dir = str(tmp_path / "indexes")
    first = index_path_for("paper:foo.pdf", base_dir=base_dir)
    second = index_path_for("paper:foo.pdf", base_dir=base_dir)
    assert first == second
