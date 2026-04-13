from collections import Counter


class NaiveTC:
    """
    Majority class classifier for technique classification.

    Predicts the most frequent technique in the training set for every
    input span regardless of content. Requires no feature extraction.
    Will achieve a non-trivial micro F1 driven entirely by the dominant
    class (Loaded Language), while achieving near-zero macro F1 across
    the remaining 13 classes.
    """

    def __init__(self):
        self.majority_class = None

    def fit(self, train_articles: list):
        """
        Determine the majority class from training articles.

        Args:
            train_articles: list of Article objects with tc_spans
        """
        counts = Counter(
            span.technique
            for article in train_articles
            for span in article.tc_spans
        )
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, spans: list) -> list:
        """
        Predict the majority class for every span.

        Args:
            spans: list of TCSpan objects (or any iterable of spans)

        Returns:
            List of predicted technique strings, one per span.
        """
        assert self.majority_class is not None, "Call fit() before predict()"
        return [self.majority_class for _ in spans]
