import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Custom k-nearest neighbors classifier function
def custom_knn(X_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        distances = []
        for train_point in X_train:
            distance = np.sqrt(np.sum((test_point - train_point) ** 2))
            distances.append(distance)
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        y_pred.append(predicted_label)
    return np.array(y_pred)

# Creating a synthetic dataset
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.5, random_state=4)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Visualizing the dataset
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='*', s=100, edgecolors='black')
plt.title("Visualized dataset", fontsize=20)
plt.show()

# Splitting the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Predicting with custom KNN classifier
y_pred_3 = custom_knn(X_train, y_train, X_test, k=3)
y_pred_1 = custom_knn(X_train, y_train, X_test, k=1)

# Calculating accuracy for custom KNN classifiers
accuracy_3 = accuracy_score(y_test, y_pred_3) * 100
accuracy_1 = accuracy_score(y_test, y_pred_1) * 100
print("Accuracy with k=3:", accuracy_3)
print("Accuracy with k=1:", accuracy_1)

# Visualizing the predictions
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_3, marker='*', s=100, edgecolors='black')
plt.title("Predicted values with k=3", fontsize=20)

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_1, marker='*', s=100, edgecolors='black')
plt.title("Predicted values with k=1", fontsize=20)
plt.show()
