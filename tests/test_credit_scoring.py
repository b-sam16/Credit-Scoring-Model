import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append("./src")
from Credit_scoring import CreditScoring  # Adjust the import path if needed

# Sample dataset for testing
@pytest.fixture
def sample_data():
    data = {
        'CustomerId': [1, 1, 2, 2, 3, 3],
        'TransactionDate': ['2024-01-01', '2024-01-10', '2024-01-02', '2024-01-15', '2024-01-03', '2024-01-12'],
        'TransactionCount': [10, 5, 7, 8, 4, 9],
        'MonetaryValue': [100, 150, 120, 130, 110, 160],
        'Default': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad'],
    }
    df = pd.DataFrame(data)
    return df

# Test the 'calculate_recency' method
def test_calculate_recency(sample_data):
    # Initialize CreditScoring class with sample data
    scoring = CreditScoring(sample_data, 'TransactionDate', 'TransactionCount', 'MonetaryValue')
    scoring.calculate_recency()
    
    # Ensure Recency is calculated properly (days since last transaction)
    recency_values = scoring.data['Recency']
    assert recency_values[0] == 9  # As the last transaction for customer 1 is on 2024-01-10, so recency should be 9 days
    assert recency_values[2] == 13  # Customer 2, last transaction on 2024-01-15, so recency is 13 days
    assert recency_values[4] == 9  # Customer 3, last transaction on 2024-01-12, so recency is 9 days

# Test the 'assign_rfms_scores' method
def test_assign_rfms_scores(sample_data):
    scoring = CreditScoring(sample_data, 'TransactionDate', 'TransactionCount', 'MonetaryValue')
    scoring.calculate_recency()  # Make sure recency is calculated first
    scoring.assign_rfms_scores('Recency', 'TransactionCount', 'MonetaryValue')
    
    # Check that RFMS scores are assigned
    assert 'Recency_Score' in scoring.data.columns
    assert 'Frequency_Score' in scoring.data.columns
    assert 'Monetary_Score' in scoring.data.columns
    assert 'RFMS_Score' in scoring.data.columns

    # Check that RFMS_Score is within expected range (0 to 4 based on quantiles)
    assert scoring.data['Recency_Score'].min() >= 1
    assert scoring.data['Recency_Score'].max() <= 4

# Test the 'assign_good_bad_labels' method
def test_assign_good_bad_labels(sample_data):
    scoring = CreditScoring(sample_data, 'TransactionDate', 'TransactionCount', 'MonetaryValue')
    scoring.calculate_recency()
    scoring.assign_rfms_scores('Recency', 'TransactionCount', 'MonetaryValue')
    
    # Assign Good/Bad labels using a threshold
    scoring.assign_good_bad_labels('RFMS_Score', 2.5)
    
    # Check that Default_Label column is added and values are either 'Good' or 'Bad'
    assert 'Default_Label' in scoring.data.columns
    assert set(scoring.data['Default_Label'].unique()).issubset({'Good', 'Bad'})

# Test the 'perform_woe_binning' method
def test_perform_woe_binning(sample_data):
    scoring = CreditScoring(sample_data, 'TransactionDate', 'TransactionCount', 'MonetaryValue')
    scoring.calculate_recency()
    scoring.assign_rfms_scores('Recency', 'TransactionCount', 'MonetaryValue')
    
    # Perform WoE binning on 'TransactionCount' and 'MonetaryValue'
    binning_result = scoring.perform_woe_binning(['TransactionCount', 'MonetaryValue'], 'Default_Label')
    
    # Check that WoE binning was performed and binning result is returned
    assert binning_result is not None  # It should return binning details
    assert 'TransactionCount_woe' in scoring.data.columns
    assert 'MonetaryValue_woe' in scoring.data.columns

# Test the 'discretize_continuous_features' method
def test_discretize_continuous_features(sample_data):
    scoring = CreditScoring(sample_data, 'TransactionDate', 'TransactionCount', 'MonetaryValue')
    scoring.calculate_recency()
    
    # Discretize continuous features
    scoring.discretize_continuous_features(['MonetaryValue'])
    
    # Check that the 'MonetaryValue_Binned' column was added
    assert 'MonetaryValue_Binned' in scoring.data.columns

# Test the 'save_to_csv' method
def test_save_to_csv(sample_data, tmp_path):
    scoring = CreditScoring(sample_data, 'TransactionDate', 'TransactionCount', 'MonetaryValue')
    scoring.calculate_recency()
    file_path = tmp_path / 'test_output.csv'
    
    # Save to CSV and ensure file is created
    scoring.save_to_csv(file_path)
    
    assert file_path.exists()
    saved_data = pd.read_csv(file_path)
    assert len(saved_data) == len(sample_data)

