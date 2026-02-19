"""Text preprocessing utility functions for the sentiment analysis project."""

import string
import nltk
from typing import List


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
STEMMER = nltk.PorterStemmer()


class TextPreprocessor:
    """Text preprocessing pipeline"""

    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def punctuation_removal(self, text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans("", "", string.punctuation))

    def tokenize(self, text: str) -> list:
        """Tokenize text into words."""
        return nltk.word_tokenize(text)

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove stop words from token list."""
        return [token for token in tokens if token not in STOP_WORDS]

    def stemming(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens."""
        return [STEMMER.stem(token) for token in tokens]

    def preprocess(self, text: str) -> List[str]:
        """Run the full preprocessing pipeline on the input text."""
        text = self.lowercase(text)
        text = self.punctuation_removal(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        tokens = self.stemming(tokens)
        return tokens
