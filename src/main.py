#authors: Oscar Paredes and Rachit ALeti
from src.preprocess import load_and_preprocess
from src.logistic_model import run_logistic
from src.knn_model import run_knn

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("\nRunning Logistic Regression...")
    run_logistic(X_train, X_test, y_train, y_test)

    print("\nRunning KNN...")
    run_knn(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()