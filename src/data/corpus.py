import os
from dataclasses import dataclass, field


@dataclass
class SISpan:
    start: int
    end: int


@dataclass
class TCSpan:
    technique: str
    start: int
    end: int


@dataclass
class Article:
    article_id: str
    text: str
    si_spans: list = field(default_factory=list)  # list[SISpan]
    tc_spans: list = field(default_factory=list)  # list[TCSpan]


def load_corpus(split: str, data_dir: str) -> list:
    """
    Load articles and their annotations for a given split.

    Args:
        split:    "train" or "dev"
        data_dir: path to the data/ directory

    Returns:
        List of Article objects. si_spans and tc_spans are empty for dev
        (no ground truth labels provided in this release).
    """
    articles_dir = os.path.join(data_dir, f"{split}-articles")
    si_labels_dir = os.path.join(data_dir, f"{split}-labels-task1-span-identification")
    tc_labels_dir = os.path.join(data_dir, f"{split}-labels-task2-technique-classification")

    articles = []

    for filename in sorted(os.listdir(articles_dir)):
        if not filename.endswith(".txt"):
            continue

        article_id = filename.replace("article", "").replace(".txt", "")
        text = _read_article(os.path.join(articles_dir, filename))
        si_spans = _read_si_labels(si_labels_dir, article_id)
        tc_spans = _read_tc_labels(tc_labels_dir, article_id)

        articles.append(Article(article_id=article_id, text=text, si_spans=si_spans, tc_spans=tc_spans))

    return articles


def _read_article(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_si_labels(labels_dir: str, article_id: str) -> list:
    path = os.path.join(labels_dir, f"article{article_id}.task1-SI.labels")
    if not os.path.exists(path):
        return []

    spans = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            spans.append(SISpan(start=int(parts[1]), end=int(parts[2])))

    return spans


def _read_tc_labels(labels_dir: str, article_id: str) -> list:
    path = os.path.join(labels_dir, f"article{article_id}.task2-TC.labels")
    if not os.path.exists(path):
        return []

    spans = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            spans.append(TCSpan(technique=parts[1], start=int(parts[2]), end=int(parts[3])))

    return spans
