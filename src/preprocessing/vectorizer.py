import numpy as np
from typing import List


class BagOfWordsVectorizer:
    """Convert preprocessed tokens into a bag-of-words vector."""

    def __init__(self):
        self.dictionary = []

    def fit(self, messages: List[List[str]]):
        """Build the dictionary from the training messages."""
        for tokens in messages:
            for token in tokens:
                if token not in self.dictionary:
                    self.dictionary.append(token)

    def transform(self, tokens: List[str]) -> np.ndarray:
        """Convert messages into bag-of-words vectors."""
        features = np.zeros(len(self.dictionary))

        for token in tokens:
            if token in self.dictionary:
                features[self.dictionary.index(token)] += 1

        return features

    def fit_transform(self, messages: List[List[str]]) -> np.ndarray:
        """Fit the vectorizer and transform the messages."""
        self.fit(messages)
        return np.array([self.transform(message) for message in messages])
