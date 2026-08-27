import os

from src.paper_registry import (index_path_for, list_registered_papers,
                                register_paper, slugify_paper_id)


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


def test_register_and_list_round_trip(tmp_path):
    base_dir = str(tmp_path / "indexes")
    register_paper("paper:foo.pdf", "foo.pdf", base_dir=base_dir)
    papers = list_registered_papers(base_dir=base_dir)
    assert len(papers) == 1
    assert papers[0]["paper_id"] == "paper:foo.pdf"
    assert papers[0]["display_name"] == "foo.pdf"


def test_register_paper_is_idempotent_per_paper_id(tmp_path):
    base_dir = str(tmp_path / "indexes")
    register_paper("paper:foo.pdf", "foo.pdf", base_dir=base_dir)
    register_paper("paper:foo.pdf", "foo (renamed).pdf", base_dir=base_dir)
    papers = list_registered_papers(base_dir=base_dir)
    assert len(papers) == 1
    assert papers[0]["display_name"] == "foo (renamed).pdf"


def test_list_registered_papers_empty_when_nothing_registered(tmp_path):
    assert list_registered_papers(base_dir=str(tmp_path / "indexes")) == []
