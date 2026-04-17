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
   - "transactions_data.csv" - make sure that you create a separate csv file full of half fraudulent and half legitimate transactions, file is too large for github
   - "train_fraud_labels.json" - make sure that you create a separate csv file full of half fraudulent and half legitimate transactions whose id's correspond to the
      transactions_data.csv, then online convert to json
3. Install dependencies: in requirements.txt
4. Create folder in computer, put in downloads
5. use streamlit to run
6. run commands on terminal
7. cd Downloads
8. cd FOLDER_NAME
9. py -m streamlit run app.py
