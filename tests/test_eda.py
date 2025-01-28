import sys
sys.path.append("../src")
import pytest
import pandas as pd
from unittest.mock import patch
import matplotlib.pyplot as plt

from src.eda import EDAHandler  # Adjust the import path based on your project structure

# Sample DataFrame for testing
@pytest.fixture
def sample_data():
    data = {
        'Age': [25, 30, 35, 40, 45],
        'Income': [50000, 60000, 70000, 80000, 90000],
        'Gender': ['M', 'F', 'M', 'F', 'M'],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    }
    return pd.DataFrame(data)

def test_overview(sample_data):
    handler = EDAHandler(sample_data)
    with patch("builtins.print") as mock_print:
        handler.overview()
        mock_print.assert_any_call("Shape (Rows, Columns): (5, 4)")
        mock_print.assert_any_call("Data Types:\nAge           int64\nIncome        int64\nGender       object\nCity         object\n")

def test_summary_statistics(sample_data):
    handler = EDAHandler(sample_data)
    with patch("builtins.print") as mock_print:
        handler.summary_statistics()
        mock_print.assert_any_call("Summary Statistics:\n       Age       Income\ncount  5.0   5.000000\nmean  35.0  70000.000000\nstd   7.91   15811.388737\nmin   25.0  50000.000000\n25%   30.0  60000.000000\n50%   35.0  70000.000000\n75%   40.0  80000.000000\nmax   45.0  90000.000000")

def test_distribution_of_numerical_features(sample_data):
    handler = EDAHandler(sample_data)
    with patch.object(plt, 'show') as mock_show:
        handler.distribution_of_numerical_features()
        mock_show.assert_called()

def test_distribution_of_categorical_features(sample_data):
    handler = EDAHandler(sample_data)
    with patch.object(plt, 'show') as mock_show:
        handler.distribution_of_categorical_features(max_unique_values=3)
        mock_show.assert_called()

def test_correlation_analysis(sample_data):
    handler = EDAHandler(sample_data)
    with patch("builtins.print") as mock_print:
        handler.correlation_analysis()
        mock_print.assert_any_call("Correlation Matrix:\n          Age  Income\nAge      1.0     1.0\nIncome   1.0     1.0")

def test_identify_missing_values(sample_data):
    handler = EDAHandler(sample_data)
    with patch("builtins.print") as mock_print:
        handler.identify_missing_values()
        mock_print.assert_any_call("Missing Values:\nAge       0\nIncome    0\nGender    0\nCity      0\n")

def test_detect_outliers(sample_data):
    handler = EDAHandler(sample_data)
    with patch.object(plt, 'show') as mock_show:
        handler.Detect_outliers()
        mock_show.assert_called()

