"""TF-IDF Vectorizer."""

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class TFIDFVectorizer:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)   # rất hợp spam detection
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save(self, file_path):
        joblib.dump(self.vectorizer, file_path)

    def load(self, file_path):
        self.vectorizer = joblib.load(file_path)
