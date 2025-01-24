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

    def convert_column_to_datetime(self, column):
        """
        Converts a column to datetime format if it is an object column with date/time values.
        """
        if self.data[column].dtype == 'object':
            self.data[column] = pd.to_datetime(self.data[column], errors='coerce')
            print(f"Column {column} converted to datetime format.")
        else:
            print(f"Column {column} is already in datetime format.")

    def ensure_absolute_value(self):
        """
        Ensure the 'Value' column is the absolute value of the 'Amount' column.
        """
        if 'Amount' in self.data.columns and 'Value' in self.data.columns:
            self.data['Value'] = self.data['Amount'].abs()
            print("Ensured 'Value' column is the absolute value of 'Amount'.")
        else:
            print("Columns 'Amount' or 'Value' are missing from the dataset.")

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

    
    def handle_outliers(self, threshold=1.5, method='cap'):
        """
        Detects and handles outliers for all numerical columns using the Interquartile Range (IQR) method.
        The 'cap' method caps the outliers within the lower and upper bounds.
        
        Parameters:
        method: str - The method to use for handling outliers ('cap')
        threshold: float - The IQR threshold to detect outliers
        """
        # Loop through all numerical columns
        for col in self.data.select_dtypes(include=np.number).columns:
            # Calculate IQR
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            if method == 'cap':
                # Cap outliers to the lower and upper bounds using the `clip` method
                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"Capped outliers in {col} to the lower and upper bounds.")
            else:
                print(f"Invalid method specified. Use 'cap'.")
