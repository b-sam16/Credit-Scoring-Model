import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureEngineering:
    def __init__(self, data):
        """
        Initialize the FeatureEngineering class.
        
        Parameters:
        data : pandas DataFrame - The dataset to perform feature engineering on
        """
        self.data = data
    
    def create_aggregate_features(self, customer_column, amount_column):
        """
        Creates aggregate features for each customer.
        
        Parameters:
        customer_column : str - The column representing the customer ID
        amount_column : str - The column representing the transaction amount
        
        Returns:
        DataFrame with aggregate features
        """
        # Aggregate features
        aggregated_data = self.data.groupby(customer_column)[amount_column].agg(
            total_transaction_amount='sum',
            average_transaction_amount='mean',
            transaction_count='count',
            std_transaction_amount='std'
        ).reset_index()
        # Merge the aggregated features back into the original dataset
        self.data = self.data.merge(aggregated_data, on=customer_column, how='left')

        return self.data
    
    def extract_features(self, date_column):
        """
        Extracts features like hour, day, month, year from the transaction datetime column.
        
        Parameters:
        date_column : str - The column representing the transaction date/time
        
        Returns:
        DataFrame with extracted features
        """
        # Convert to datetime format if it's not already
        self.data[date_column] = pd.to_datetime(self.data[date_column], errors='coerce')

        # Extract features if the conversion is successful
        if self.data[date_column].isnull().sum() == 0:
            self.data['Transaction_Year'] = self.data[date_column].dt.year
            self.data['Transaction_Month'] = self.data[date_column].dt.month
            self.data['Transaction_Day'] = self.data[date_column].dt.day
            self.data['Transaction_Hour'] = self.data[date_column].dt.hour
            print("Date features extracted successfully.")
        else:
            print(f"Conversion failed. Some values in {date_column} could not be parsed.")
    
    def encode_categorical_variables(self, categorical_columns, encoding_method='onehot'):
        """
        Encodes categorical variables using One-Hot Encoding or Label Encoding.
        
        Parameters:
        categorical_columns : list - List of columns to encode
        encoding_method : str - The encoding method: 'onehot' or 'label'
        
        Returns:
        DataFrame with encoded categorical variables
        """
        if encoding_method == 'onehot':
            encoder = OneHotEncoder(drop='first', sparse_output=False)
            encoded_df = pd.DataFrame(encoder.fit_transform(self.data[categorical_columns]))
            encoded_df.columns = encoder.get_feature_names_out(categorical_columns)
            self.data = pd.concat([self.data, encoded_df], axis=1)
            self.data = self.data.drop(columns=categorical_columns)
        elif encoding_method == 'label':
            encoder = LabelEncoder()
            for col in categorical_columns:
                self.data[col] = encoder.fit_transform(self.data[col])
        return self.data
    
    def handle_missing_values(self, columns, strategy='mean'):
        """
        Handle missing values by imputation if any missing values are found.
        
        Parameters:
        columns : list - List of columns with missing values to handle
        strategy : str - The imputation strategy: 'mean', 'median', 'most_frequent'
        
        Returns:
        DataFrame with missing values handled
        """
        # Check if there are any missing values in the specified columns
        missing_columns = [col for col in columns if self.data[col].isnull().sum() > 0]
        
        if not missing_columns:
            print("No missing values found in the specified columns.")
            return self.data
        
        # Apply imputation only to the columns with missing values
        print(f"Imputing missing values for columns: {missing_columns} using strategy: {strategy}")
        imputer = SimpleImputer(strategy=strategy)
        self.data[missing_columns] = imputer.fit_transform(self.data[missing_columns])
        
        print(f"Missing values imputed for columns: {missing_columns}")
        return self.data
    
    def normalize_standardize(self, columns, method='standardize', return_full_df=True):
        """
        Normalize or standardize numerical features.
        
        Parameters:
        columns : list - List of columns to normalize/standardize
        method : str - The method to use: 'normalize' or 'standardize'
        
        Returns:
        DataFrame with normalized/standardized features
        """
        if method == 'normalize':
            scaler = MinMaxScaler()
            suffix = '_normalized'
        elif method == 'standardize':
            scaler = StandardScaler()
            suffix = '_standardized'
        else:
            raise ValueError("Invalid method. Please choose 'normalize' or 'standardize'.")
        
        #Applyscaling and rename columns
        scaled_data = scaler.fit_transform(self.data[columns])
        scaled_columns = [col + suffix for col in columns]

        # Add the scaled columns to the DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_columns, index=self.data.index)
        self.data = pd.concat([self.data, scaled_df], axis=1)
    
        if return_full_df:
            return self.data
        else:
            return scaled_df
        