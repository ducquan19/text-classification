"""
Model evaluation and experiment tracking module.

This module provides functionality to:
- Evaluate multiple models on the same dataset
- Store experiment results for comparison
- Generate comparison reports
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.vectorizer import BagOfWordsVectorizer

VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0


class ModelEvaluator:
    """Evaluator for comparing multiple models on the same dataset."""

    def __init__(self, dataset_path: str, results_dir: str = "experiment_results"):
        """
        Initialize the evaluator.

        Args:
            dataset_path: Path to the CSV dataset file
            results_dir: Directory to store experiment results
        """
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.preprocessor = TextPreprocessor()
        self.vectorizer = BagOfWordsVectorizer()
        self.label_encoder = LabelEncoder()

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

    def load_data(self):
        """Load dataset from CSV file."""
        df = pd.read_csv(self.dataset_path)
        messages = df["Message"].values.tolist()
        labels = df["Category"].values.tolist()
        return messages, labels

    def prepare_data(
        self, messages: List[str], labels: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess and split the dataset.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in messages]

        # Vectorize
        X = self.vectorizer.fit_transform(preprocessed_texts)

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=TEST_SIZE, shuffle=True, random_state=SEED
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def evaluate_model(
        self,
        model,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Train and evaluate a single model.

        Args:
            model: Model instance with train() and predict() methods
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Label sets
            model_name: Name of the model for logging

        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Training {model_name}...")
        print(f"{'=' * 60}")

        # Train the model
        start_time = datetime.now()
        model.train(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate metrics for each dataset split
        results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "training_time_seconds": training_time,
            "dataset_sizes": {
                "train": len(y_train),
                "val": len(y_val),
                "test": len(y_test),
            },
            "metrics": {
                "train": self._calculate_metrics(y_train, y_train_pred),
                "val": self._calculate_metrics(y_val, y_val_pred),
                "test": self._calculate_metrics(y_test, y_test_pred),
            },
            "confusion_matrix": {
                "test": confusion_matrix(y_test, y_test_pred).tolist()
            },
            "classification_report": {
                "test": classification_report(
                    y_test,
                    y_test_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True,
                )
            },
        }

        # Print summary
        print(f"\n{model_name} Results:")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Train Accuracy: {results['metrics']['train']['accuracy']:.4f}")
        print(f"Val Accuracy:   {results['metrics']['val']['accuracy']:.4f}")
        print(f"Test Accuracy:  {results['metrics']['test']['accuracy']:.4f}")
        print(f"Test F1-Score:  {results['metrics']['test']['f1_weighted']:.4f}")

        return results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate performance metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def save_results(
        self, results: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """
        Save experiment results to JSON file.

        Args:
            results: Results dictionary
            filename: Optional custom filename

        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results["model_name"].replace(" ", "_").lower()
            filename = f"{model_name}_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {filepath}")
        return filepath

    def compare_models(self, result_files: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare results from multiple experiments.

        Args:
            result_files: List of result file paths. If None, loads all JSON files in results_dir

        Returns:
            DataFrame with comparison metrics
        """
        if result_files is None:
            result_files = [
                os.path.join(self.results_dir, f)
                for f in os.listdir(self.results_dir)
                if f.endswith(".json")
            ]

        if not result_files:
            print("No result files found for comparison.")
            return pd.DataFrame()

        # Load all results
        all_results = []
        for filepath in result_files:
            with open(filepath, "r", encoding="utf-8") as f:
                result = json.load(f)
                all_results.append(result)

        # Create comparison dataframe
        comparison_data = []
        for result in all_results:
            row = {
                "Model": result["model_name"],
                "Timestamp": result["timestamp"],
                "Training Time (s)": result["training_time_seconds"],
                "Train Size": result["dataset_sizes"]["train"],
                "Val Size": result["dataset_sizes"]["val"],
                "Test Size": result["dataset_sizes"]["test"],
                "Train Accuracy": result["metrics"]["train"]["accuracy"],
                "Val Accuracy": result["metrics"]["val"]["accuracy"],
                "Test Accuracy": result["metrics"]["test"]["accuracy"],
                "Test Precision": result["metrics"]["test"]["precision_weighted"],
                "Test Recall": result["metrics"]["test"]["recall_weighted"],
                "Test F1": result["metrics"]["test"]["f1_weighted"],
            }
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by test accuracy descending
        comparison_df = comparison_df.sort_values("Test Accuracy", ascending=False)

        return comparison_df

    def generate_comparison_report(
        self, comparison_df: pd.DataFrame, output_file: Optional[str] = None
    ) -> str:
        """
        Generate and save a comparison report.

        Args:
            comparison_df: DataFrame from compare_models()
            output_file: Optional output file path

        Returns:
            Path to the saved report
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.results_dir, f"comparison_report_{timestamp}.txt"
            )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of models compared: {len(comparison_df)}\n\n")

            f.write("-" * 80 + "\n")
            f.write("RANKING BY TEST ACCURACY\n")
            f.write("-" * 80 + "\n\n")

            for rank, (idx, row) in enumerate(comparison_df.iterrows(), start=1):
                f.write(f"Rank #{rank}: {row['Model']}\n")
                f.write(f"  Test Accuracy:  {row['Test Accuracy']:.4f}\n")
                f.write(f"  Test F1:        {row['Test F1']:.4f}\n")
                f.write(f"  Test Precision: {row['Test Precision']:.4f}\n")
                f.write(f"  Test Recall:    {row['Test Recall']:.4f}\n")
                f.write(f"  Training Time:  {row['Training Time (s)']:.2f}s\n")
                f.write(f"  Timestamp:      {row['Timestamp']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("DETAILED COMPARISON TABLE\n")
            f.write("-" * 80 + "\n\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            # Best model summary
            best_model = comparison_df.iloc[0]
            f.write("-" * 80 + "\n")
            f.write("BEST MODEL SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Model: {best_model['Model']}\n")
            f.write(f"Test Accuracy: {best_model['Test Accuracy']:.4f}\n")
            f.write(f"Test F1 Score: {best_model['Test F1']:.4f}\n")
            f.write(f"Training Time: {best_model['Training Time (s)']:.2f}s\n\n")

        print(f"\nComparison report saved to: {output_file}")
        return output_file

    def run_experiment(
        self, models: List[Tuple[Any, str]], save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run a complete experiment with multiple models.

        Args:
            models: List of (model_instance, model_name) tuples
            save_results: Whether to save results to files

        Returns:
            List of result dictionaries
        """
        print("\n" + "=" * 60)
        print("STARTING EXPERIMENT")
        print("=" * 60)
        print(f"Dataset: {self.dataset_path}")
        print(f"Number of models: {len(models)}")
        print(f"Results directory: {self.results_dir}")

        # Load and prepare data once
        messages, labels = self.load_data()
        print(f"\nDataset loaded: {len(messages)} samples, {len(set(labels))} classes")

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
            messages, labels
        )
        print(f"Data split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

        # Evaluate each model
        all_results = []
        for model, model_name in models:
            results = self.evaluate_model(
                model, X_train, X_val, X_test, y_train, y_val, y_test, model_name
            )
            all_results.append(results)

            if save_results:
                self.save_results(results)

        # Generate comparison
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE - GENERATING COMPARISON")
        print("=" * 60)

        comparison_df = self.compare_models()
        print("\n" + comparison_df.to_string(index=False))

        if save_results:
            self.generate_comparison_report(comparison_df)
            self.plot_comparison_results(comparison_df, all_results)

        return all_results

    def plot_comparison_results(
        self, comparison_df: pd.DataFrame, all_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate visualization plots for model comparison.

        Args:
            comparison_df: DataFrame with comparison metrics
            all_results: List of result dictionaries

        Returns:
            Path to the saved plot
        """
        if comparison_df.empty:
            print("No data to plot.")
            return ""

        # Set style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16, 10))

        # Create 2x2 subplot layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Test Metrics Comparison (bar chart)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics_data = comparison_df[
            ["Model", "Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
        ]
        metrics_melted = metrics_data.melt(
            id_vars=["Model"], var_name="Metric", value_name="Score"
        )
        metrics_melted["Metric"] = metrics_melted["Metric"].str.replace("Test ", "")

        sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric", ax=ax1)
        ax1.set_title("Test Set Performance Comparison", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Model", fontsize=10)
        ax1.set_ylabel("Score", fontsize=10)
        ax1.set_ylim(0.8, 1.0)
        ax1.legend(title="Metric", loc="lower right")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Training Time Comparison (bar chart)
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(
            data=comparison_df,
            x="Model",
            y="Training Time (s)",
            ax=ax2,
            palette="viridis",
        )
        ax2.set_title("Training Time Comparison", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Model", fontsize=10)
        ax2.set_ylabel("Time (seconds)", fontsize=10)
        ax2.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for container in ax2.containers:
            ax2.bar_label(container, fmt="%.2f")

        # 3. Train vs Val vs Test Accuracy (grouped bar)
        ax3 = fig.add_subplot(gs[1, 0])
        accuracy_data = comparison_df[
            ["Model", "Train Accuracy", "Val Accuracy", "Test Accuracy"]
        ]
        accuracy_melted = accuracy_data.melt(
            id_vars=["Model"], var_name="Split", value_name="Accuracy"
        )
        accuracy_melted["Split"] = accuracy_melted["Split"].str.replace(" Accuracy", "")

        sns.barplot(
            data=accuracy_melted,
            x="Model",
            y="Accuracy",
            hue="Split",
            ax=ax3,
            palette="Set2",
        )
        ax3.set_title(
            "Accuracy Across Train/Val/Test Splits", fontsize=12, fontweight="bold"
        )
        ax3.set_xlabel("Model", fontsize=10)
        ax3.set_ylabel("Accuracy", fontsize=10)
        ax3.set_ylim(0.8, 1.0)
        ax3.legend(title="Dataset Split")
        ax3.tick_params(axis="x", rotation=45)

        # 4. Confusion Matrix for Best Model
        ax4 = fig.add_subplot(gs[1, 1])
        best_result = all_results[0]  # First result is best after sorting
        cm = np.array(best_result["confusion_matrix"]["test"])

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax4, cbar_kws={"label": "Count"}
        )
        ax4.set_title(
            f'Confusion Matrix - {best_result["model_name"]} (Best Model)',
            fontsize=12,
            fontweight="bold",
        )
        ax4.set_xlabel("Predicted Label", fontsize=10)
        ax4.set_ylabel("True Label", fontsize=10)
        ax4.set_xticklabels(self.label_encoder.classes_)
        ax4.set_yticklabels(self.label_encoder.classes_)

        # Overall title
        fig.suptitle("Model Comparison Results", fontsize=16, fontweight="bold", y=0.98)

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.results_dir, f"comparison_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nVisualization saved to: {plot_path}")
        return plot_path
