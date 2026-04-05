#authors: Oscar Paredes and Rachit Aleti
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def run_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n*** KNN Results ***")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, predictions))