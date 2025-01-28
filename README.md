

# Credit Scoring Project  

## Overview  
The **Credit Scoring Project** aims to develop a robust machine learning model for predicting credit risk and scoring customers based on their likelihood of default. This project is built for **Bati Bank**, which collaborates with an eCommerce company to enable a **Buy-Now-Pay-Later** (BNPL) service. The solution includes building, deploying, and serving credit scoring predictions.  

## Objectives  
- Build a machine learning model to predict the likelihood of loan default.  
- Assign credit scores to customers based on risk probabilities.  
- Incorporate **Weight of Evidence (WOE) binning** for feature engineering using **scorecardpy**.  
- Ensure scalability, modularity, and maintainability.  

---  

## Features  
1. **Exploratory Data Analysis (EDA)**:  
   - Data summarization and visualization.  
   - Detecting and handling outliers.  
   - Analyzing correlations and trends.  

2. **Feature Engineering**:  
   - **Weight of Evidence (WOE)** binning using **scorecardpy** to transform categorical and numerical features.  
   - Handling missing values and scaling features.  
   - Creating new derived features to improve model performance.  

3. **Model Development**:  
   - Built using machine learning models like Random Forest and Gradient Boosting.  
   - Hyperparameter tuning with GridSearchCV and RandomizedSearchCV.  
   - Evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.  

4. **Model Deployment**:  
   - Deployment-ready with REST API for real-time predictions.  
   - Models are serialized using **joblib** for easy integration.  

5. **Reproducibility**:  
   - Well-documented workflows for data preparation, feature engineering, and model training.  
   - Unit testing ensures pipeline reliability.  

---  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/credit_scoring_project.git  
   cd credit_scoring_project  
   ```  

2. Create and activate a virtual environment:  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate  
   ```  

3. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

---  

## Usage  
1. Run the data preparation pipeline:  
   ```bash  
   python src/feature_engineering.py  
   ```  

2. Train the machine learning models:  
   ```bash  
   python src/model_training.py  
   ```  

3. Save the trained models:  
   ```bash  
   python scripts/main.py --save-model  
   ```  

---  

## Key Libraries Used  
- **Pandas**: For data manipulation and analysis.  
- **scikit-learn**: For building and evaluating machine learning models.  
- **scorecardpy**: For WOE binning and credit scoring.  
- **joblib**: For saving and loading models.  
- **FastAPI**: For building the REST API.  


---  
