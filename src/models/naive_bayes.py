"""Naive Bayes model for sentiment analysis."""

from sklearn.naive_bayes import MultinomialNB
import joblib


class NaiveBayesClassifier:
    """A Naive Bayes model for sentiment analysis."""

    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)
