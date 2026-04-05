#authors: Oscar Paredes and Rachit Aleti
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def train_models(X_train, y_train):
    log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    return log_model, knn_model