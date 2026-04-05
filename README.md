# Financial Risk Predictor (Fraud Detection)

## Authors
- Oscar Paredes
- Rachit Aleti

## Overview
This project uses machine learning to predict whether a financial transaction is fraudulent based on transaction behavior.

## Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)

## Dataset
Kaggle Financial Transactions Fraud Dataset

## Features
- Transaction attributes (amount, type, etc.)
- Time-based features (hour, day of week)
- Encoded categorical variables

## Results
Models are evaluated using:
- Precision
- Recall
- F1 Score
- Confusion Matrix

## How to Run
This project uses the *Financial Transactions Fraud Dataset* from Kaggle:
https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets
1. Clone repository
2. Place these 2 dataset files in `/data`
   - "transactions_data.csv"
   - "train_fraud_labels.json"
3. Install dependencies: in requirements.txt
4. Run project
