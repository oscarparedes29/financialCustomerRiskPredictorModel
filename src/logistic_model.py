#authors: OScar Paredes and Rachit Aleti
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def run_logistic(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n*** Logistic Regression Results ***")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, predictions))