import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import joblib


class CreditScoringModel:
    def __init__(self, data, feature_columns, target_column):
        """
        Initialize the CreditScoringModel class.
        """
        self.data = data
        
        # Drop target column from the feature columns
        feature_columns = [col for col in feature_columns if col != target_column]
        
        # Ensure the target column is separate from the features
        self.X = data[feature_columns]
        self.y = data[target_column]
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Initialize models with default parameters
        self.rf_model = RandomForestClassifier(random_state=42)
        self.gbm_model = GradientBoostingClassifier(random_state=42)
        
        # Placeholder for best models
        self.rf_best_model = None
        self.gbm_best_model = None

    def train(self):
        """
        Train the Random Forest and Gradient Boosting models.
        """
        # Cross-validation for Random Forest
        rf_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"Random Forest Cross-Validation Accuracy: {rf_scores.mean():.4f}")

        # Train Random Forest
        self.rf_model.fit(self.X_train, self.y_train)

        # Cross-validation for Gradient Boosting
        gbm_scores = cross_val_score(self.gbm_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"Gradient Boosting Cross-Validation Accuracy: {gbm_scores.mean():.4f}")

        # Train Gradient Boosting
        self.gbm_model.fit(self.X_train, self.y_train)

    def save_model(self, model, model_name):
        """
        Save the trained model to a file.
        """
        filename = f"{model_name}_model.joblib"
        joblib.dump(model, filename)
        print(f"Model saved as {filename}")

    def load_model(self, model_name):
        """
        Load a trained model from a file.
        """
        filename = f"{model_name}_model.joblib"
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model

    def evaluate(self, model, model_name='Model'):
        """
        Evaluate the model using various metrics.
        """
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"Confusion Matrix for {model_name}:\n", cm)

        # Classification Report
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(self.y_test, y_pred))

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc_value = auc(fpr, tpr)
        
        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc='lower right')
        plt.show()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

    def compare_models(self):
        """
        Compare both models (Random Forest and GBM) on test data.
        """
        print("\nEvaluating Random Forest...")
        rf_metrics = self.evaluate(self.rf_model, 'Random Forest')

        print("\nEvaluating Gradient Boosting...")
        gbm_metrics = self.evaluate(self.gbm_model, 'Gradient Boosting')

        print("\nRandom Forest Evaluation Metrics:")
        print(rf_metrics)
        
        print("\nGradient Boosting Evaluation Metrics:")
        print(gbm_metrics)

    def tune_hyperparameters(self):
        """
        Perform hyperparameter tuning for both Random Forest and Gradient Boosting.
        """
        # Random Forest Hyperparameter Tuning
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
        }
        rf_grid_search = GridSearchCV(self.rf_model, rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        rf_grid_search.fit(self.X_train, self.y_train)
        self.rf_best_model = rf_grid_search.best_estimator_
        print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")

        # Gradient Boosting Hyperparameter Tuning
        gbm_param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }
        gbm_grid_search = GridSearchCV(self.gbm_model, gbm_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        gbm_grid_search.fit(self.X_train, self.y_train)
        self.gbm_best_model = gbm_grid_search.best_estimator_
        print(f"Best Gradient Boosting Parameters: {gbm_grid_search.best_params_}")

    def get_best_model(self):
        """
        Return the best models after hyperparameter tuning.
        """
        if self.rf_best_model and self.gbm_best_model:
            return self.rf_best_model, self.gbm_best_model
        else:
            print("Hyperparameter tuning has not been performed yet.")
            return None, None
