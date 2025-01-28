import pandas as pd
import numpy as np
import scorecardpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

class CreditScoring:
    def __init__(self, data, recency_column, frequency_column, monetary_column):
        """
        Initialize the CreditScoring class with the dataset and the necessary columns.
        
        Parameters:
        data : pandas DataFrame - The dataset for default estimation and WoE binning.
        recency_column : str - The column representing the transaction date.
        frequency_column : str - The column representing the transaction count or frequency.
        monetary_column : str - The column representing the monetary value of transactions.
        """
        self.data = data
        self.recency_column = recency_column
        self.frequency_column = frequency_column
        self.monetary_column = monetary_column
    
    def calculate_recency(self):
        """
        Calculate Recency as the number of days since the last transaction.
        Also assigns Frequency and Monetary values from the respective columns.
        """
        # Ensure the recency column is in datetime format
        self.data[self.recency_column] = pd.to_datetime(self.data[self.recency_column])
        
        # Calculate Recency as the number of days since the latest transaction
        # Calculate Recency (days since the last transaction per customer)
        self.data['Recency'] = self.data.groupby('CustomerId')[self.recency_column].transform(lambda x: (x.max() - x).dt.days)

        # Assign Frequency and Monetary columns
        self.data['Frequency'] = self.data[self.frequency_column]
        self.data['Monetary'] = self.data[self.monetary_column]
        
        return self.data
    
    def assign_rfms_scores(self, recency_col, frequency_col, monetary_col):
        """
        Assign scores to Recency, Frequency, and Monetary values using quantiles.
        
        Parameters:
        recency_col : str - The column representing Recency
        frequency_col : str - The column representing Frequency
        monetary_col : str - The column representing Monetary
        
        Returns:
        DataFrame with RFMS scores assigned
        """
        # Recency (lower is better, so reverse the score)
        self.data['Recency_Score'] = pd.qcut(self.data[recency_col], 4, labels=[4, 3, 2, 1]).astype(int)

        # Frequency (higher is better)
        self.data['Frequency_Score'] = pd.qcut(self.data[frequency_col], 4, labels=[1, 2, 3, 4]).astype(int)

        # Monetary (higher is better)
        self.data['Monetary_Score'] = pd.qcut(self.data[monetary_col], 4, labels=[1, 2, 3, 4]).astype(int)

        # RFMS Score (weighted average, weights can be adjusted)
        self.data['RFMS_Score'] = (
            self.data['Recency_Score'] * 0.25 +
            self.data['Frequency_Score'] * 0.35 +
            self.data['Monetary_Score'] * 0.40
        )
        return self.data

    def assign_good_bad_labels(self, rfms_col, threshold):
        """
        Assign Good and Bad labels based on RFMS scores.
        
        Parameters:
        rfms_col : str - The column representing RFMS Score
        threshold : float - The threshold to classify Good (high RFMS) and Bad (low RFMS)
        
        Returns:
        DataFrame with Good/Bad labels
        """
        self.data['Default_Label'] = np.where(self.data[rfms_col] >= threshold,'Good', 'Bad')  
        return self.data

    def perform_woe_binning(self, feature_cols, target_col):
        """
        Perform Weight of Evidence (WoE) binning on the specified features.

        Parameters:
        feature_cols : list - List of feature column names for WoE binning.
        target_col : str - Target column name for binning.
        """
        try:

            # Map the target column to numeric values (1 for Good, 0 for Bad)
            self.data[target_col] = self.data[target_col].map({'Good': 1, 'Bad': 0})
        
            # Ensure no missing values after mapping
            if self.data[target_col].isnull().any():
                raise ValueError("Target column contains invalid values that cannot be mapped.")

            # WoE binning using scorecardpy's woebin function
            binning = sc.woebin(self.data, y=target_col, x=feature_cols)

            # Transform the dataset to include WoE-transformed values
            transformed_data = sc.woebin_ply(self.data, binning)

            # Keep original columns and append WoE-transformed columns
            for feature in feature_cols:
                transformed_data.rename(columns={feature: f'{feature}_woe'}, inplace=True)

            # Concatenate the original data with the transformed data
            self.data = pd.concat([self.data, transformed_data[[f'{feature}_woe' for feature in feature_cols]]], axis=1)

            print("WoE binning completed successfully.")
            return binning  # Return binning details for inspection, if needed
        except Exception as e:
            print(f"An error occurred during WoE binning: {e}")


    def discretize_continuous_features(self, cols_to_bin, bins=5):
        """
        Discretize continuous features into bins for better modeling.
        
        Parameters:
        cols_to_bin : list - List of continuous feature columns to discretize
        bins : int - Number of bins to create
        
        Returns:
        DataFrame with discretized features
        """
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        for col in cols_to_bin:
            self.data[col + '_Binned'] = discretizer.fit_transform(self.data[[col]])
        return self.data

    def plot_rfms_space(self, recency_col, frequency_col):
        """
        Plot the RFMS space (Recency vs Frequency).
        
        Parameters:
        recency_col : str - The column representing Recency
        frequency_col : str - The column representing Frequency
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data[recency_col], y=self.data[frequency_col], hue=self.data['Default_Label'], palette='coolwarm', s=100, alpha=0.7)
        plt.title('RFMS Space: Recency vs Frequency')
        plt.xlabel('Recency')
        plt.ylabel('Frequency')
        handles, _ = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles, title='Default', labels=['Good', 'Bad'])
        plt.show()

    def plot_monetary_vs_frequency(self, monetary_col, frequency_col):
        """
        Plot Monetary vs Frequency.
        
        Parameters:
        monetary_col : str - The column representing Monetary
        frequency_col : str - The column representing Frequency
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data[monetary_col], y=self.data[frequency_col], hue=self.data['Default_Label'], palette='coolwarm', s=100, alpha=0.7)
        plt.title('Monetary vs Frequency')
        plt.xlabel('Monetary')
        plt.ylabel('Frequency')
        handles, _ = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles, title='Default', labels=['Good', 'Bad'])
        plt.show()

    def plot_rfm_distribution(self):
        """Plot the distribution of Recency, Frequency, and Monetary."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sns.histplot(self.data['Recency'], kde=True, ax=axes[0], color='skyblue', bins=30)
        axes[0].set_title('Recency Distribution')
        axes[0].set_xlabel('Recency (days)')
        axes[0].set_ylabel('Frequency')
        
        sns.histplot(self.data['Frequency'], kde=True, ax=axes[1], color='lightgreen', bins=30)
        axes[1].set_title('Frequency Distribution')
        axes[1].set_xlabel('Frequency')
        axes[1].set_ylabel('Frequency')
        
        sns.histplot(self.data['Monetary'], kde=True, ax=axes[2], color='salmon', bins=30)
        axes[2].set_title('Monetary Distribution')
        axes[2].set_xlabel('Monetary')
        axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_rfms_score_distribution(self):
        """
        Plot the distribution of RFMS scores.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['RFMS_Score'], kde=True, bins=20, color='skyblue')
        plt.title('Distribution of RFMS Scores')
        plt.xlabel('RFMS Score')
        plt.ylabel('Frequency')
        plt.show()

    def plot_correlation_heatmap(self, recency_col, frequency_col, monetary_col):
        """
        Plot a heatmap of correlations between Recency, Frequency, and Monetary.
        
        Parameters:
        recency_col : str - The column representing Recency
        frequency_col : str - The column representing Frequency
        monetary_col : str - The column representing Monetary
        """
        correlation_matrix = self.data[[recency_col, frequency_col, monetary_col]].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap: Recency, Frequency, and Monetary')
        plt.show()

    def plot_woe_binning(self, feature_cols):
        """
        Plot WoE binning for the specified features.
        
        Parameters:
        feature_cols : list - List of feature columns to plot WoE binning
        """
        for feature in feature_cols:
            woe_feature = feature + '_woe'  # Adjusting to match the column names with '_woe'
        
            # Ensure the column exists before attempting to plot
            if woe_feature in self.data.columns:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=self.data[woe_feature], y=self.data[woe_feature], palette='coolwarm')
                plt.title(f'Weight of Evidence (WoE) for {feature}')
                plt.xlabel('WoE')
                plt.ylabel(feature)
                plt.show()
            else:
                print(f"Column {woe_feature} not found in the DataFrame.")

    def save_to_csv(self, file_path):
        """
        Save the processed DataFrame to a CSV file.
    
        Parameters:
        file_path : str - The path to the file where the DataFrame will be saved.
    """
        try:
            self.data.to_csv(file_path, index=False)
            print(f"Data saved successfully to {file_path}")
        except Exception as e:
            print(f"An error occurred while saving the data: {e}")
