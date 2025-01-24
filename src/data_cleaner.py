import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataCleaner:
    def __init__(self, data):
        """
        Initialize the EDAHandler class with a pandas DataFrame.
        """
        self.data = data

    def handle_missing_values(self, strategy='mean', fill_value=None):
        """
        Handle missing values in the dataset.
        Args:
        - strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', or 'constant').
        - fill_value (any): Value to fill missing values if strategy is 'constant'.
        """
        # Check if there are any missing values in the dataset
        if self.data.isnull().sum().sum() == 0:
            print("No missing values detected in the dataset.")
            return

        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:  # Check for missing values
                if self.data[col].dtype in ['float64', 'int64']:  # Numerical columns
                    if strategy == 'mean':
                        self.data[col].fillna(self.data[col].mean(), inplace=True)
                    elif strategy == 'median':
                        self.data[col].fillna(self.data[col].median(), inplace=True)
                    elif strategy == 'mode':
                        self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                    elif strategy == 'constant':
                        self.data[col].fillna(fill_value, inplace=True)
                    else:
                        raise ValueError(f"Invalid strategy {strategy} for numerical column {col}.")
                elif self.data[col].dtype == 'object':  # Categorical columns
                    if strategy == 'mode':
                        self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                    elif strategy == 'constant':
                        self.data[col].fillna(fill_value if fill_value is not None else 'Unknown', inplace=True)
                    else:
                        raise ValueError(f"Invalid strategy {strategy} for categorical column {col}.")
        print("Missing values handled successfully!")

    
    def detect_outliers(self, column, threshold=1.5):
        """
        Detects outliers using the Interquartile Range (IQR) method.
        Returns a mask of outliers in the column.
        """
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (self.data[column] < lower_bound) | (self.data[column] > upper_bound)
        return outliers
    
    def handle_outliers(self, column, method='cap', lower_cap=None, upper_cap=None):
        """
        Handle outliers in a column.
        Methods: 'cap' (capping), 'remove' (removing outliers)
        
        Parameters:
        method: str - The method to use for handling outliers ('cap' or 'remove')
        lower_cap: float - Lower threshold for capping
        upper_cap: float - Upper threshold for capping
        """
        if method == 'remove':
            # Remove outliers by filtering them out
            outliers = self.detect_outliers(column)
            self.data = self.data[~outliers]
            print(f"Removed {outliers.sum()} outliers from {column}")
        
        elif method == 'cap':
            # Cap the values outside the range to the threshold (if thresholds are provided)
            if lower_cap is not None:
                self.data[column] = np.where(self.data[column] < lower_cap, lower_cap, self.data[column])
            if upper_cap is not None:
                self.data[column] = np.where(self.data[column] > upper_cap, upper_cap, self.data[column])
            print(f"Capped outliers in {column} to specified thresholds.")
        else:
            print("Invalid method specified. Use 'cap' or 'remove'.")


