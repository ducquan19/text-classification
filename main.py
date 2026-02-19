"""
Text Classification - Spam/Ham Predictor

A command-line tool for training and predicting spam/ham messages
using various machine learning models.
"""

import argparse
import sys
import os
import joblib
import pandas as pd
from typing import Dict, Any

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.vectorizer import BagOfWordsVectorizer
from src.models.naive_bayes import NaiveBayesClassifier
from src.models.knn import KNNClassifier
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.svm import SVMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Constants
MODELS_DIR = "trained_models"
DATASET_PATH = "dataset/data.csv"
SEED = 0


def print_guide():
    """Print usage guide."""
    guide = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                   TEXT CLASSIFICATION - USER GUIDE                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

AVAILABLE MODELS:
  • naive_bayes      - Naive Bayes Classifier (fast, good baseline)
  • knn              - K-Nearest Neighbors (k=5)
  • logistic         - Logistic Regression (high accuracy)
  • svm              - Support Vector Machine (robust)

USAGE EXAMPLES:

  1. Train a model:
     python main.py --model naive_bayes --train

  2. Predict with a trained model:
     python main.py --model naive_bayes --predict "Free money now!"

  3. Interactive prediction mode:
     python main.py --model logistic --interactive

  4. Show this guide:
     python main.py --guide

  5. Compare all models:
     python run_comparison.py

COMMANDS:
  --guide              Show this usage guide
  --train              Train the specified model
  --predict TEXT       Predict classification for given text
  --interactive        Enter interactive mode for multiple predictions
  --model MODEL        Choose model (default: naive_bayes)
  --force-retrain      Force retraining even if model exists

EXAMPLES OF TEXT TO CLASSIFY:
  • "Can we meet tomorrow at 3pm?"  (likely: ham)
  • "FREE iPhone! Click now!"       (likely: spam)
  • "I'll call you later"           (likely: ham)
  • "Win $1000 cash prize now!"     (likely: spam)

╚═══════════════════════════════════════════════════════════════════════════╝
"""
    print(guide)


def get_model_instance(model_name: str):
    """Get model instance by name."""
    models = {
        "naive_bayes": NaiveBayesClassifier(),
        "knn": KNNClassifier(n_neighbors=5),
        "logistic": LogisticRegressionClassifier(),
        "svm": SVMClassifier(),
    }
    return models.get(model_name)


def train_model(model_name: str, force_retrain: bool = False):
    """Train and save a model."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_filename = f"{model_name}_model.pkl"
    vectorizer_filename = f"{model_name}_vectorizer.pkl"
    encoder_filename = f"{model_name}_encoder.pkl"

    model_path = os.path.join(MODELS_DIR, model_filename)
    vectorizer_path = os.path.join(MODELS_DIR, vectorizer_filename)
    encoder_path = os.path.join(MODELS_DIR, encoder_filename)

    # Check if model already exists
    if not force_retrain and os.path.exists(model_path):
        print(
            f"✓ Model '{model_name}' already trained. Use --force-retrain to retrain."
        )
        return

    print(f"\n{'=' * 60}")
    print(f"Training {model_name.upper()} model...")
    print(f"{'=' * 60}\n")

    # Load data
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    messages = df["Message"].values.tolist()
    labels = df["Category"].values.tolist()
    print(f"✓ Loaded {len(messages)} samples")

    # Preprocess
    print("Preprocessing data...")
    preprocessor = TextPreprocessor()
    preprocessed = [preprocessor.preprocess(msg) for msg in messages]
    print("✓ Preprocessing complete")

    # Vectorize
    print("Vectorizing data...")
    vectorizer = BagOfWordsVectorizer()
    X = vectorizer.fit_transform(preprocessed)
    print(f"✓ Created vectors with {len(vectorizer.dictionary)} features")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    print(f"✓ Encoded labels: {label_encoder.classes_}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"✓ Train size: {len(y_train)}, Test size: {len(y_test)}")

    # Train model
    print(f"\nTraining {model_name} model...")
    model = get_model_instance(model_name)
    model.train(X_train, y_train)
    print("✓ Model training complete")

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).sum() / len(y_test)
    print(f"✓ Test Accuracy: {accuracy:.4f}")

    # Save model
    print("\nSaving model...")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Vectorizer saved to {vectorizer_path}")
    print(f"✓ Encoder saved to {encoder_path}")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}\n")


def load_model(model_name: str) -> Dict[str, Any]:
    """Load trained model, vectorizer, and encoder."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, f"{model_name}_vectorizer.pkl")
    encoder_path = os.path.join(MODELS_DIR, f"{model_name}_encoder.pkl")

    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model '{model_name}' not found!")
        print(f"   Train it first: python main.py --model {model_name} --train\n")
        sys.exit(1)

    print(f"Loading {model_name} model...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    encoder = joblib.load(encoder_path)
    print("✓ Model loaded successfully\n")

    return {
        "model": model,
        "vectorizer": vectorizer,
        "encoder": encoder,
        "preprocessor": TextPreprocessor(),
    }


def predict_text(text: str, model_name: str):
    """Predict classification for given text."""
    # Load model
    components = load_model(model_name)
    model = components["model"]
    vectorizer = components["vectorizer"]
    encoder = components["encoder"]
    preprocessor = components["preprocessor"]

    # Preprocess and vectorize
    preprocessed = preprocessor.preprocess(text)
    vector = vectorizer.transform(preprocessed).reshape(1, -1)

    # Predict
    prediction_encoded = model.predict(vector)[0]
    prediction = encoder.inverse_transform([prediction_encoded])[0]

    # Display result
    emoji = "🚫" if prediction == "spam" else "✅"
    print(f"{'=' * 60}")
    print(f"Text: {text}")
    print(f"Prediction: {emoji} {prediction.upper()}")
    print(f"{'=' * 60}\n")

    return prediction


def interactive_mode(model_name: str):
    """Interactive prediction mode."""
    print(f"\n{'=' * 60}")
    print(f"INTERACTIVE MODE - {model_name.upper()}")
    print(f"{'=' * 60}")
    print("Enter text to classify (or 'quit' to exit)\n")

    # Load model once
    components = load_model(model_name)
    model = components["model"]
    vectorizer = components["vectorizer"]
    encoder = components["encoder"]
    preprocessor = components["preprocessor"]

    while True:
        try:
            text = input("Enter text: ").strip()

            if text.lower() in ["quit", "exit", "q"]:
                print("\nExiting interactive mode. Goodbye! 👋\n")
                break

            if not text:
                print("⚠ Please enter some text.\n")
                continue

            # Preprocess and predict
            preprocessed = preprocessor.preprocess(text)
            vector = vectorizer.transform(preprocessed).reshape(1, -1)
            prediction_encoded = model.predict(vector)[0]
            prediction = encoder.inverse_transform([prediction_encoded])[0]

            # Display result
            emoji = "🚫" if prediction == "spam" else "✅"
            print(f"→ {emoji} {prediction.upper()}\n")

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode. Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Text Classification - Spam/Ham Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--guide", action="store_true", help="Show usage guide")

    parser.add_argument(
        "--model",
        type=str,
        default="naive_bayes",
        choices=["naive_bayes", "knn", "logistic", "svm"],
        help="Choose model (default: naive_bayes)",
    )

    parser.add_argument("--train", action="store_true", help="Train the model")

    parser.add_argument(
        "--predict",
        type=str,
        metavar="TEXT",
        help="Predict classification for given text",
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Enter interactive prediction mode"
    )

    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if model exists",
    )

    args = parser.parse_args()

    # Show guide if requested or no arguments
    if args.guide or len(sys.argv) == 1:
        print_guide()
        return

    # Train model
    if args.train:
        train_model(args.model, args.force_retrain)
        return

    # Predict single text
    if args.predict:
        predict_text(args.predict, args.model)
        return

    # Interactive mode
    if args.interactive:
        interactive_mode(args.model)
        return

    # No action specified
    print("\n⚠ No action specified. Use --guide to see usage instructions.\n")


if __name__ == "__main__":
    main()
