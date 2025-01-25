import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDAHandler:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def overview(self):
        """Understand the structure of the dataset: rows, columns, and data types."""
        print(f"Data Overview:")
        print(f"Shape (Rows, Columns): {self.data.shape}")
        print(f"\nData Types:\n{self.data.dtypes}")

        
    def summary_statistics(self):
        """Display summary statistics for the dataset."""
        print(f"Summary Statistics:\n{self.data.describe()}")
        
    def distribution_of_numerical_features(self):
        """Visualize the distribution of numerical features to identify patterns, skewness, and potential outliers."""
        numerical_columns = self.data.select_dtypes(include=['number']).columns

        if numerical_columns.empty:
            print("No numerical columns to visualize.")
            return
        
        for col in numerical_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.data[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.show()

    def distribution_of_categorical_features(self,max_unique_values=25):
        """
        Plot bar charts for all categorical columns in the dataset,
        excluding those with one unique value or high cardinality (too many unique values).
        """
        # Select categorical columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
    
        for col in categorical_cols:
            # Skip columns with only one unique value
            if self.data[col].nunique() == 1:
                print(f"Skipping {col} - Only one unique value")
                continue
        
            # Skip columns with more than max_unique_values unique values
            if self.data[col].nunique() > max_unique_values:
                print(f"Skipping {col} - Too many unique values ({self.data[col].nunique()})")
                continue  
        
        
            plt.figure(figsize=(10, 6))
            value_counts = self.data[col].value_counts().head(10)  # Show top 10 categories
            sns.barplot(x=value_counts.index, y=value_counts.values, color='skyblue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

    def correlation_analysis(self):
        """
        Plot the correlation matrix of numerical columns.
        """
        numerical_cols = self.data.select_dtypes(include=['number']).columns

        if len(numerical_cols) < 2:
            print("Not enough numerical columns to compute a correlation matrix.")
            return

        correlation_matrix = self.data[numerical_cols].corr()
        print("Correlation Matrix:\n", correlation_matrix)  # Print the correlation matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def identify_missing_values(self):
        """Check for missing values and visualize the missing data pattern."""
        missing_values = self.data.isnull().sum()
        print(f"Missing Values:\n{missing_values[missing_values > 0]}")
        

    def Detect_outliers(self):
        """
        Plot box plots to detect outliers in numerical columns.
        """
        numerical_cols = self.data.select_dtypes(include=['number']).columns

        if numerical_cols.empty:
            print("No numerical columns to visualize outliers.")
            return

        for col in numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.data[col])
            plt.title(f"Box Plot of {col}")
            plt.xlabel(col)
            plt.show()
