"""Training pipeline.

Example usage:
    # Using default Naive Bayes
    trainer = Trainer()

    # Using KNN
    from src.models.knn import KNNClassifier
    trainer = Trainer(model=KNNClassifier(n_neighbors=3))

    # Using Logistic Regression
    from src.models.logistic_regression import LogisticRegressionClassifier
    trainer = Trainer(model=LogisticRegressionClassifier())

    # Using SVM
    from src.models.svm import SVMClassifier
    trainer = Trainer(model=SVMClassifier())
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.vectorizer import BagOfWordsVectorizer
from src.models.naive_bayes import NaiveBayesClassifier

VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0


class Trainer:
    """Trainer class to orchestrate the training pipeline.

    Supports multiple model types including:
    - NaiveBayesClassifier
    - KNNClassifier
    - LogisticRegressionClassifier
    - SVMClassifier
    - and any model that implements train() and predict() methods
    """

    def __init__(self, model=None):
        """Initialize the trainer with a specified model.

        Args:
            model: A classifier instance (e.g., NaiveBayesClassifier, KNNClassifier).
                   If None, defaults to NaiveBayesClassifier.
        """
        self.preprocessor = TextPreprocessor()
        self.vectorizer = BagOfWordsVectorizer()
        self.label_encoder = LabelEncoder()
        self.model = model if model is not None else NaiveBayesClassifier()

    def train(self, texts, labels):
        """Train the model on the provided texts and labels."""
        # Preprocess the texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]

        # Vectorize the preprocessed texts
        X = self.vectorizer.fit_transform(preprocessed_texts)

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=TEST_SIZE, shuffle=True, random_state=SEED
        )

        print("Start training...")
        self.model.train(X_train, y_train)
        print("Training completed!")

        # evaluate
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)

        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"Val accuracy: {val_acc}")
        print(f"Test accuracy: {test_acc}")

        return self.model, self.vectorizer, self.label_encoder
