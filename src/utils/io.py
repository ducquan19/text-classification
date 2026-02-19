"""Input/Output utility functions for the sentiment analysis project."""

import pandas as pd

def load_dataset(path) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    return pd.read_csv(path)
