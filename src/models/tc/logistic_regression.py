from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class LogisticRegressionTC:
    """
    Multi-class technique classifier using TF-IDF span representations.

    Each propaganda span is represented as a TF-IDF vector, where each
    dimension corresponds to a word in the vocabulary and its value reflects
    how distinctive that word is to the span relative to the rest of the
    corpus. A multi-class logistic regression classifier is trained using a
    one-vs-rest strategy, learning a separate set of weights for each of the
    14 technique classes. Class weighting is applied to partially compensate
    for the severe class imbalance in the training set.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """
        Args:
            C:        inverse regularisation strength for LR
            max_iter: maximum iterations for LR solver
        """
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=50000)
        self.clf = LogisticRegression(
            class_weight='balanced',
            C=C,
            max_iter=max_iter,
            solver='lbfgs',
        )

    def fit(self, train_articles: list):
        """
        Fit the vectoriser and classifier on training articles.

        Args:
            train_articles: list of Article objects with tc_spans
        """
        spans_text, labels = self._extract_spans(train_articles)
        X = self.vectorizer.fit_transform(spans_text)
        self.clf.fit(X, labels)
        return self

    def predict(self, articles: list) -> list:
        """
        Predict technique labels for all TC spans in each article.

        Args:
            articles: list of Article objects with tc_spans

        Returns:
            List of lists of predicted technique strings,
            one list per article, in the same order as article.tc_spans.
        """
        predictions = []

        for article in articles:
            if not article.tc_spans:
                predictions.append([])
                continue

            spans_text = [
                article.text[span.start:span.end]
                for span in article.tc_spans
            ]
            X = self.vectorizer.transform(spans_text)
            predictions.append(list(self.clf.predict(X)))

        return predictions

    def predict_flat(self, articles: list) -> list:
        """
        Return a flat list of predictions across all articles,
        aligned with a flat list of gold labels for evaluation.

        Args:
            articles: list of Article objects

        Returns:
            List of predicted technique strings.
        """
        return [
            label
            for article_preds in self.predict(articles)
            for label in article_preds
        ]

    # ------------------------------------------------------------------

    def _extract_spans(self, articles: list):
        """Extract flat lists of span texts and labels from articles."""
        spans_text = []
        labels = []
        for article in articles:
            for span in article.tc_spans:
                spans_text.append(article.text[span.start:span.end])
                labels.append(span.technique)
        return spans_text, labels
