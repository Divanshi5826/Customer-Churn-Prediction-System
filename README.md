# Customer Churn Prediction System

An end-to-end customer churn analytics and prediction system built using Python, Pandas, Scikit-learn, and Streamlit.

## 🚀 Live Demo

**Streamlit Dashboard:**
https://customer-churn-prediction-system-nrfwkqrd4apypenqh7qnb6.streamlit.app/

## 📌 Overview

Customer Churn Prediction System analyzes customer demographic, subscription, service usage, and billing data to identify patterns associated with customer churn.

The project combines exploratory data analysis, business insights, machine learning, and an interactive dashboard to help identify customers who may be at higher risk of leaving a service.

## 🎯 Objectives

* Analyze customer churn behavior
* Clean and preprocess real-world customer data
* Identify important factors associated with churn
* Compare machine learning classification models
* Provide business-oriented insights
* Build an interactive churn prediction dashboard

## 📊 Dataset

The project uses the Telco Customer Churn dataset containing **7,043 customer records** and information related to:

* Demographics
* Customer tenure
* Phone and internet services
* Contract information
* Payment methods
* Monthly charges
* Total charges
* Customer churn status

After cleaning invalid `TotalCharges` values, **7,032 records** were used for analysis and modeling.

## 🔄 Project Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Business Analysis
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Feature Importance Analysis
      ↓
Interactive Streamlit Dashboard
      ↓
Customer Churn Prediction
```

## 🔍 Data Analytics

The project analyzes churn patterns across:

* Contract type
* Payment method
* Internet service
* Customer tenure
* Monthly charges
* Customer demographics
* Service subscriptions

## 🤖 Machine Learning Models

### Logistic Regression

Used as a baseline classification model.

### Random Forest Classifier

Used as the final predictive model for capturing nonlinear relationships between customer characteristics and churn behavior.

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## 📈 Feature Importance

Random Forest feature importance is used to identify the variables that contribute most to the model's predictions.

The analysis helps identify customer characteristics that are useful for understanding churn risk.

## 💡 Business Insights

The analysis highlights differences in churn behavior across contract types, payment methods, tenure groups, internet service categories, and monthly charges.

These insights can help businesses identify higher-risk customer segments and develop targeted retention strategies.

## 🖥️ Interactive Dashboard

The Streamlit dashboard provides four major sections:

### 1. Overview

Displays:

* Total customers
* Churned customers
* Overall churn rate
* Average monthly charges
* Average customer tenure

### 2. Churn Analysis

Interactive analysis of churn patterns across different customer attributes.

### 3. Churn Prediction

Users can enter customer details and receive:

* Churn prediction
* Churn probability
* Customer risk indication

### 4. Model Performance

Displays the project's modeling methodology and evaluation information.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib
* Jupyter Notebook

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│
├── churn_prediction.ipynb
├── app.py
├── feature_importance.png
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## 🔮 Future Improvements

* Hyperparameter tuning
* Gradient Boosting / XGBoost comparison
* Advanced customer segmentation
* Automated model monitoring
* Database integration
* Real-time data pipelines
