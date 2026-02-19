import numpy as np

from src.preprocessing.text_preprocessor import TextPreprocessor


class Predictor:

    def __init__(self, model, vectorizer, label_encoder):

        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.preprocessor = TextPreprocessor()

    def predict(self, text: str):

        processed = self.preprocessor.preprocess(text)

        features = self.vectorizer.transform(processed)
        features = np.array(features).reshape(1, -1)

        pred = self.model.predict(features)

        label = self.label_encoder.inverse_transform(pred)[0]

        return label


