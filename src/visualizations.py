import seaborn as sns
import matplotlib.pyplot as plt

class DataVisualizer:
    """
    A class for visualizing various aspects of a dataset,
    including distributions, categorical features, correlations, and outliers.
    """

    def __init__(self, data):
        """
        Initialize the visualizer with a dataset.
        
        Parameters:
            data (pd.DataFrame): The dataset to visualize.
        """
        self.data = data
        self.numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    def plot_numerical_distribution(self):
        """
        Plot the distribution of numerical columns using histograms and KDE.
        """
        if not self.numerical_cols:
            print("No numerical columns to visualize.")
            return

        for col in self.numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def plot_categorical_distribution(self,max_unique_values=25):
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
            
    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix of numerical columns.
        """
        if len(self.numerical_cols) < 2:
            print("Not enough numerical columns to compute a correlation matrix.")
            return

        correlation_matrix = self.data[self.numerical_cols].corr()
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def plot_outliers(self):
        """
        Plot box plots to detect outliers in numerical columns.
        """
        if not self.numerical_cols:
            print("No numerical columns to visualize outliers.")
            return

        for col in self.numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.data[col])
            plt.title(f"Box Plot of {col}")
            plt.xlabel(col)
            plt.show()
