import os
import pytest
from src.data.corpus import load_corpus
from src.data.splits import make_splits

DATA_DIR = '/Users/kippbatchelor/Documents/UNI/uni_y3/FYP/FYP_work_folder/data'


@pytest.fixture(scope='module')
def splits():
    articles = load_corpus('train', DATA_DIR)
    return make_splits(articles, seed=42)


def test_splits_sum_to_total(splits):
    train, dev, test = splits
    assert len(train) + len(dev) + len(test) == 371


def test_splits_are_disjoint(splits):
    train, dev, test = splits
    train_ids = {a.article_id for a in train}
    dev_ids   = {a.article_id for a in dev}
    test_ids  = {a.article_id for a in test}
    assert len(train_ids & dev_ids)  == 0
    assert len(train_ids & test_ids) == 0
    assert len(dev_ids   & test_ids) == 0


def test_splits_are_deterministic():
    articles = load_corpus('train', DATA_DIR)
    train1, dev1, test1 = make_splits(articles, seed=42)
    train2, dev2, test2 = make_splits(articles, seed=42)
    assert [a.article_id for a in train1] == [a.article_id for a in train2]
    assert [a.article_id for a in dev1]   == [a.article_id for a in dev2]
    assert [a.article_id for a in test1]  == [a.article_id for a in test2]


def test_different_seeds_give_different_splits():
    articles = load_corpus('train', DATA_DIR)
    train1, _, _ = make_splits(articles, seed=42)
    train2, _, _ = make_splits(articles, seed=99)
    assert [a.article_id for a in train1] != [a.article_id for a in train2]


def test_approximate_split_sizes(splits):
    train, dev, test = splits
    assert 270 <= len(train) <= 310
    assert 30  <= len(dev)   <= 50
    assert 30  <= len(test)  <= 50
