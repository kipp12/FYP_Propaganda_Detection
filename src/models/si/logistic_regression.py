from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.bio import tokenize_words, char_offsets_to_word_bio, word_bio_to_char_offsets


class LogisticRegressionSI:
    """
    Token-level BIO sequence labeller for span identification.

    Each token is represented by a TF-IDF feature vector derived from a
    fixed window of surrounding tokens, capturing local lexical context.
    The classifier assigns a BIO tag to each token independently.
    Class weighting is applied to address the severe token-level imbalance
    between propaganda and non-propaganda tokens.

    Predicted BIO sequences are converted back to character offsets for
    evaluation.
    """

    def __init__(self, window_size: int = 2, C: float = 1.0, max_iter: int = 5000):
        """
        Args:
            window_size: number of tokens either side of the target token
                         to include in the feature window (default 2)
            C:           inverse regularisation strength for LR
            max_iter:    maximum iterations for LR solver
        """
        self.window_size = window_size
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
            train_articles: list of Article objects with si_spans
        """
        windows, labels = self._build_features(train_articles, training=True)
        X = self.vectorizer.fit_transform(windows)
        self.clf.fit(X, labels)
        return self

    def predict(self, articles: list) -> list:
        """
        Predict propaganda spans for a list of articles.

        Args:
            articles: list of Article objects

        Returns:
            List of lists of (start, end) character offset tuples,
            one list per article.
        """
        predictions = []

        for article in articles:
            tokens = tokenize_words(article.text)
            if not tokens:
                predictions.append([])
                continue

            words = [w for w, _, _ in tokens]
            windows = [self._window_string(words, i) for i in range(len(words))]

            X = self.vectorizer.transform(windows)
            tags = self.clf.predict(X)

            tagged_tokens = [
                (word, tag, start, end)
                for (word, start, end), tag in zip(tokens, tags)
            ]
            predictions.append(word_bio_to_char_offsets(tagged_tokens))

        return predictions

    # ------------------------------------------------------------------

    def _build_features(self, articles: list, training: bool):
        """Build window strings and BIO labels for all tokens in all articles."""
        windows = []
        labels = []

        for article in articles:
            tagged = char_offsets_to_word_bio(article.text, article.si_spans)
            words = [w for w, _, _, _ in tagged]

            for i, (_, tag, _, _) in enumerate(tagged):
                windows.append(self._window_string(words, i))
                if training:
                    labels.append(tag)

        return windows, labels

    def _window_string(self, words: list, idx: int) -> str:
        """Return a single string of tokens within the window around idx."""
        start = max(0, idx - self.window_size)
        end = min(len(words), idx + self.window_size + 1)
        return ' '.join(words[start:end])
