"""
Example script demonstrating how to use the ModelEvaluator
to compare different classification models.
"""

from src.training.evaluator import ModelEvaluator
from src.models.naive_bayes import NaiveBayesClassifier
from src.models.knn import KNNClassifier
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.svm import SVMClassifier


def main():
    # Initialize evaluator
    evaluator = ModelEvaluator(
        dataset_path="dataset/data.csv", results_dir="experiment_results"
    )

    # Define models to compare
    models = [
        (NaiveBayesClassifier(), "Naive Bayes"),
        (KNNClassifier(n_neighbors=3), "KNN (k=3)"),
        (KNNClassifier(n_neighbors=5), "KNN (k=5)"),
        (KNNClassifier(n_neighbors=7), "KNN (k=7)"),
        (LogisticRegressionClassifier(), "Logistic Regression"),
        (SVMClassifier(), "SVM"),
    ]

    # Run experiment
    print("Running comparison experiment across multiple models...")
    results = evaluator.run_experiment(models, save_results=True)

    # The results are automatically saved and a comparison report is generated
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("Check the 'experiment_results' directory for:")
    print("  - Individual model results (.json files)")
    print("  - Comparison report (.txt file)")
    print("  - Comparison visualization (.png file)")
    print("=" * 60)


if __name__ == "__main__":
    main()
