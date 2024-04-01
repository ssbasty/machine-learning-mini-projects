import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Building KNN Classifier objects
knn3 = KNeighborsClassifier(n_neighbors=3)
knn1 = KNeighborsClassifier(n_neighbors=1)

# Fitting the models
knn3.fit(X_train, y_train)
knn1.fit(X_train, y_train)

# Getting predictions for KNN classifiers
y_pred_3 = knn3.predict(X_test)
y_pred_1 = knn1.predict(X_test)

# Predicting accuracy for KNN classifiers
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


