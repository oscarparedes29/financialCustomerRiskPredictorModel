#initial creation
#authors: Oscar Paredes and Rachit Aleti

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess():
    transactions = pd.read_csv("data/transactions_data.csv")
    fraud_labels = pd.read_json("data/train_fraud_labels.json")
    df = transactions.merge(fraud_labels, on="transaction_id")

    # Time features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.dayofweek
    df = df.drop(columns=["transaction_id", "timestamp"], errors="ignore")

    # Encode categorical
    categorical_cols = df.select_dtypes(include=["object"]).columns

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, le_dict, X.columns
