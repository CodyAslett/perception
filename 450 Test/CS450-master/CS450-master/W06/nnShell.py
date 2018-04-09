"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""

from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn import datasets as sk_dataset
from sklearn.neural_network import MLPClassifier as mlp
from iris import get_iris
from get_diabetes import get_diabetes

def cs450shell(algorithm):
    try:
        print("Neural Network Algorithm - CS 450 BYU-Idaho")
        print("Which dataset would you like to use?")
        print("1 - Iris dataset")
        print("2 - Pima Indians dataset")

        nnDataset = int(input("> "))

        if (nnDataset < 1 or nnDataset > 2):
            raise Exception("invalid dataset selection. Please try again.")
        elif (nnDataset == 1 or nnDataset == 2):
            executeAlgorithm(algorithm, nnDataset)
        else:
            raise Exception("unknown error occurred")
    except (ValueError) as err:
        print("ERROR: {}".format(err))

def executeAlgorithm(algorithm, dataset):
    classifier = None
    model = None

    if (dataset == 1):
        data, targets, classes = get_iris()
    elif (dataset == 2):
        data, targets, classes = get_diabetes()

    count = 0

    #split dataset into random parts
    train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=.3)

    if (algorithm == 7):
        classifier = NeuralNetwork()
        model = classifier.fit(train_data, train_target, classes)
    else:
        classifier = mlp()
        model = classifier.fit(train_data, train_target)

    #target_predicted is an array of predictions that is received by the predict
    target_predicted = model.predict(test_data)

    #loop through the target_predicted and count up the correct predictions
    for index in range(len(target_predicted)):
        #increment counter for every match from
        #target_predicted and test_target
        if target_predicted[index] == test_target[index]:
            count += 1

    accuracy = get_accuracy(count, len(test_data))

    print("Accuracy: {:.2f}%".format(accuracy))

# get_accuracy()
#
# This function calculates and returns the accuracy
def get_accuracy(count, length):
    return (count / length) * 100