import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocess import load_and_preprocess
from src.models import train_models

st.title("Financial Risk Predictor")

@st.cache_resource
def load_everything():
    X_train, X_test, y_train, y_test, scaler, le_dict, feature_names = load_and_preprocess()

    # use a large sample but cap to avoid crashing
    max_rows = 100000
    if len(X_train) > max_rows:
        X_train = X_train[:max_rows]
        y_train = y_train[:max_rows]

    log_model, knn_model = train_models(X_train, y_train)
    return X_test, y_test, scaler, feature_names, log_model, knn_model

X_test, y_test, scaler, feature_names, log_model, knn_model = load_everything()

model_choice = st.selectbox("Model", ["Logistic Regression", "KNN"])

if model_choice == "Logistic Regression":
    preds = log_model.predict(X_test)
else:
    # limit KNN prediction size (KNN is slow)
    max_test = 20000
    X_test_small = X_test[:max_test]
    y_test_small = y_test[:max_test]
    preds = knn_model.predict(X_test_small)
    y_test = y_test_small

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": preds
})

results["Correct"] = results["Actual"] == results["Predicted"]

st.subheader("Results Table")
st.dataframe(results.head(50))

correct_count = results["Correct"].sum()
incorrect_count = len(results) - correct_count

fig, ax = plt.subplots()
ax.bar(["Correct", "Incorrect"], [correct_count, incorrect_count])
ax.set_title("Prediction Accuracy Breakdown")

st.subheader("Prediction Graph")
st.pyplot(fig)