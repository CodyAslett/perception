"""
Author: Cutler Hollist
Purpose: This program adds to the "experiment shell" where the author
    will load training and test data, implement a hard-coded classifier
    as a placeholder, and report results with that classifier.
    This assignment is primarily meant to act as a framework for
    future assignments.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from slkearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import datasets
#from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class HardCodedClassifier:
    def fit(self, data, targets):
        return HardCodedModel()


class HardCodedModel:    
    def predict(self, test_data):
        predictions = []
        for i in range(len(test_data)):
            predictions.append(0)
        return predictions


class HollistKNNClassifier:
    def __init__(self, k=1):
        self.k = k

    def fit(self, k, data, targets):
        return HollistKNNModel(k, data, targets)


class HollistKNNModel:
    def __init__(self, k, data, targets):
        self.k = k
        self.data = data
        self.targets = targets

    def predict(self, k, test_data, t=targets, train_data=data):
        for n in range(test_data):
            distances = np.sum((test_data - train_data[n,:]) ** 2, axis = 1)
            predictions = np.zeros(test_data)

            indices = np.argsort(distances, axis = 0)
            classes = np.unique(t[indices[:k]])
            if len(classes) == 1:
                predictions[n] = np.unique(classes)
            else:
                counts = np.zeros(stats.mode(classes) + 1 )
                for i in range(k):
                    counts[t[indices[i]]] += 1
                predictions[n] = stats.mode(counts)
        return predictions

    #method predict takes test_data, train_data, k:
    # for each test_data:
    #   distances = distances between test_data and train_data
    #   selection = k smallest train_targets from collection
    #   prediction = mode(selection)
    # return prediction


#def knn(k, data, dataClass, inputs):
#    nInputs = np.shape(inputs)[0]
#    closest = np.zeros(nInputs)
            
#    for n in range(nInputs):
        # Compute distances
#        distances = np.sum((data - inputs[n,:]) ** 2, axis = 1)

        # Identify the nearest neighbours 
#        indices = np.argsort(distances, axis = 0)
                
#        classes = np.unique(dataClass[indices[:k]])
#        if len(classes) == 1:
#            closest[n] = np.unique(classes)
#        else:
#            counts = np.zeros(max(classes) + 1 )
#            for i in range(k):
#                counts[dataClass[indices[i]]] += 1
#            closest[n] = np.max(counts)
            
#    return closest

# Load data
iris = datasets.load_iris()

# Prepare Training/Test Sets
# Iris
iris_train, iris_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size = 0.3)
# Newsgroups
#newsgroups_train = fetch_20newsgroups.(subset='train')
#newsgroups_train_files = newsgroups_train.filenames
#newsgroups_train_targets = newsgroups_train.target
#newsgroups_test = fetch_20newsgroups.(subset='test')
#newsgroups_test_files = newsgroups_test.filenames
#newsgroups_test_targets = newsgroups_test.target

# Create a model
classifier1 = GaussianNB()
classifier2 = HardCodedClassifier()
classifier3 = HollistKNNClassifier(3)
classifier4 = KNeighborsClassifier(n_neighbors=3)
model1 = classifier1.fit(iris_train, targets_train)
model2 = classifier2.fit(iris_train, targets_train)
model3 = classifier3.fit(iris_train, targets_train)
model4 = classifier4.fit(iris_train, targets_train)


# Make predictions
targets_predicted1 = model1.predict(iris_test)
targets_predicted2 = model2.predict(iris_test)
targets_predicted3 = model3.predict(iris_test)
targets_predicted4 = model4.predict(iris_test)

accuracy1 = 0
accuracy2 = 0
accuracy3 = 0
accuracy4 = 0
for i in range(len(targets_test)):
    if targets_predicted1[i] == targets_test[i]:
        accuracy1 = accuracy1 + 1
    if targets_predicted2[i] == targets_test[i]:
        accuracy2 = accuracy2 + 1
    if targets_predicted3[i] == targets_test[i]:
        accuracy3 = accuracy3 + 1
    if targets_predicted4[i] == targets_test[i]:
        accuracy4 = accuracy4 + 1
accuracy1 = (accuracy1 / len(targets_test)) * 100
accuracy2 = (accuracy2 / len(targets_test)) * 100
accuracy3 = (accuracy3 / len(targets_test)) * 100
accuracy4 = (accuracy4 / len(targets_test)) * 100
print("Gaussian Accuracy:" + str(accuracy1) + "%")
print("HardCodedModel Accuracy:" + str(accuracy2) + "%")
print("HollistKNN Accuracy:" + str(accuracy3) + "%")
print("KNeighbors Accuracy:" + str(accuracy4) + "%")


def main():
    return 0


if __name__ == "__main__":
    main()