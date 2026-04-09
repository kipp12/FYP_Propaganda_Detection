import random
from collections import Counter


def make_splits(articles: list, dev_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """
    Split articles into train/dev/test sets.

    Stratified by each article's dominant propaganda technique so that rare
    techniques are represented across all three splits. Articles with no
    TC spans are treated as their own stratum.

    Args:
        articles:   list of Article objects (from load_corpus)
        dev_ratio:  proportion of data for dev set  (default 0.1)
        test_ratio: proportion of data for test set (default 0.1)
        seed:       random seed for reproducibility

    Returns:
        (train, dev, test) as lists of Article objects
    """
    rng = random.Random(seed)

    # Group articles by dominant technique
    strata = {}
    for article in articles:
        key = _dominant_technique(article)
        strata.setdefault(key, []).append(article)

    train, dev, test = [], [], []

    for key, group in strata.items():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * test_ratio))
        n_dev = max(1, round(n * dev_ratio))
        # Ensure we don't exceed group size
        n_test = min(n_test, n)
        n_dev = min(n_dev, n - n_test)

        test.extend(group[:n_test])
        dev.extend(group[n_test:n_test + n_dev])
        train.extend(group[n_test + n_dev:])

    return train, dev, test


def _dominant_technique(article) -> str:
    """Return the most frequent technique in an article, or 'none' if no spans."""
    if not article.tc_spans:
        return "none"
    counts = Counter(span.technique for span in article.tc_spans)
    return counts.most_common(1)[0][0]
