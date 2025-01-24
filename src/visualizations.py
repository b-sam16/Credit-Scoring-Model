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

    def plot_categorical_distribution(self):
        """
        Plot the distribution of categorical columns using bar plots.
        """
        if not self.categorical_cols:
            print("No categorical columns to visualize.")
            return

        for col in self.categorical_cols:
            plt.figure(figsize=(8, 4))
            self.data[col].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
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
