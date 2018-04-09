"""
Author: Cutler Hollist
Purpose: This program is an "experiment shell" where the author will
    load training and test data, implement a hard-coded classifier
    as a placeholder, and report results with that classifier.
    This assignment is primarily meant to act as a framework for
    future assignments.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

class HardCodedModel:    
    def predict(self, test_data):
        predictions = []
        for i in range(len(test_data)):
            predictions.append(0)
        return predictions


class HardCodedClassifier:
    def fit(self, data, targets):
        return HardCodedModel()


# Load data
iris = datasets.load_iris()

# Prepare Training/Test Sets
data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size = 0.3)

# Create a model
classifier1 = GaussianNB()
classifier2 = HardCodedClassifier()
model1 = classifier1.fit(data_train, targets_train)
model2 = classifier2.fit(data_train, targets_train)

# Make predictions
targets_predicted1 = model1.predict(data_test)
targets_predicted2 = model2.predict(data_test)

accuracy1 = 0
accuracy2 = 0
for i in range(len(targets_test)):
    if targets_predicted1[i] == targets_test[i]:
        accuracy1 = accuracy1 + 1
    if targets_predicted2[i] == targets_test[i]:
        accuracy2 = accuracy2 + 1
accuracy1 = (accuracy1 / len(targets_test)) * 100
accuracy2 = (accuracy2 / len(targets_test)) * 100
print("Gaussian Accuracy:" + str(accuracy1) + "%")
print("HardCodedModel Accuracy:" + str(accuracy2) + "%")


def main():
    return 0


if __name__ == "__main__":
    main()