class NaiveSI:
    """
    All-O predictor for span identification.

    Assigns the non-propaganda label to every token in every article
    regardless of content. Requires no training and produces no predicted
    spans, so recall is zero by definition and F1 is therefore zero.
    Serves as a clean lower bound — any model that cannot beat this has
    failed to learn anything meaningful.
    """

    def predict(self, articles: list) -> list:
        """
        Args:
            articles: list of Article objects

        Returns:
            List of empty span lists — one per article.
        """
        return [[] for _ in articles]
