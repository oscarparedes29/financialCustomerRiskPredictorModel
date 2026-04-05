#authors: Oscar Paredes and Rachit Aleti
import streamlit as st
import pandas as pd
from src.preprocess import load_and_preprocess
from src.models import train_models

st.title("Financial Risk Predictor")
st.write("Predict whether a transaction is fraudulent")

X_train, X_test, y_train, y_test, scaler, le_dict, feature_names = load_and_preprocess()
log_model, knn_model = train_models(X_train, y_train)

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "KNN"]
)
st.subheader("Enter Transaction Details")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    if model_choice == "Logistic Regression":
        prediction = log_model.predict(input_scaled)[0]
    else:
        prediction = knn_model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("Fraudulent Transaction Detected.")
    else:
        st.success("Legitimate Transaction.")