import os
import pytest
from src.data.corpus import load_corpus, Article, SISpan, TCSpan

DATA_DIR = '/Users/kippbatchelor/Documents/UNI/uni_y3/FYP/FYP_work_folder/data'


def test_load_train_returns_correct_count():
    articles = load_corpus('train', DATA_DIR)
    assert len(articles) == 371


def test_load_dev_returns_correct_count():
    articles = load_corpus('dev', DATA_DIR)
    assert len(articles) == 75


def test_articles_are_article_objects():
    articles = load_corpus('train', DATA_DIR)
    assert all(isinstance(a, Article) for a in articles)


def test_article_has_text():
    articles = load_corpus('train', DATA_DIR)
    assert all(len(a.text) > 0 for a in articles)


def test_si_spans_are_sispan_objects():
    articles = load_corpus('train', DATA_DIR)
    for a in articles:
        assert all(isinstance(s, SISpan) for s in a.si_spans)


def test_tc_spans_are_tcspan_objects():
    articles = load_corpus('train', DATA_DIR)
    for a in articles:
        assert all(isinstance(s, TCSpan) for s in a.tc_spans)


def test_span_offsets_are_within_text():
    articles = load_corpus('train', DATA_DIR)
    for a in articles:
        for s in a.si_spans:
            assert 0 <= s.start < s.end <= len(a.text)


def test_dev_articles_have_no_labels():
    articles = load_corpus('dev', DATA_DIR)
    assert all(len(a.si_spans) == 0 for a in articles)
    assert all(len(a.tc_spans) == 0 for a in articles)
