# Titanic Survival Prediction (Machine Learning Project)

This project predicts Titanic passenger survival using a Random Forest model.  
It demonstrates data cleaning, feature engineering, pipeline modeling, evaluation, and model deployment.

## 📦 Features Included
- Cleaned missing values in `Age`, `Embarked`
- Dropped irrelevant columns (`Cabin`, `Ticket`, etc.)
- Created a custom feature: `risk_category`
- One-hot encoded categorical variables
- Built a Scikit-Learn pipeline with:
  - `StandardScaler`
  - `RandomForestClassifier`
- Evaluated model using:
  - Confusion Matrix
  - Classification Report
  - AUC Score

## 🧠 Model Performance
- **Accuracy**: ~81%
- **AUC Score**: ~0.86
- Balanced precision & recall

## 💾 Files Included
- `titanic_model_script.py` – full project code
- `model.pkl` – trained model saved with `joblib`
- `predictions.csv` – optional output CSV for clients (if uploaded)

## 🧪 Tools Used
- Python
- pandas
- scikit-learn
- joblib

## 🚀 Author
Created by [AdamML34](https://github.com/AdamML34) – aspiring ML engineer & freelancer
