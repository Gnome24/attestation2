from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


datasets = [make_classification(n_samples=np.random.randint(100, 200), n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1) for _ in range(3)]

for i, (X, y) in enumerate(datasets):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_accuracy = []
    test_accuracy = []

    for k in range(1, 9):
      
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

      
        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)

       
        train_accuracy.append(accuracy_score(y_train, y_train_pred))
        test_accuracy.append(accuracy_score(y_test, y_test_pred))

        if k == 8:
            print(f"Dataset {i+1}")
            print("Precision: ", precision_score(y_test, y_test_pred))
            print("Recall: ", recall_score(y_test, y_test_pred))
            print("F1-score: ", f1_score(y_test, y_test_pred))
            print("Confusion Matrix: ")
            print(confusion_matrix(y_test, y_test_pred))

    
    plt.figure()
    plt.plot(range(1, 9), train_accuracy, label='Train Accuracy')
    plt.plot(range(1, 9), test_accuracy, label='Test Accuracy')
    plt.legend()
    plt.title(f"Dataset {i+1}")
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.show()
