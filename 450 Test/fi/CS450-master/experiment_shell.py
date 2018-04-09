"""
CS 450 Experiment Shell
Author: Scott Burton

This program is intended to be an experiment shell for testing machine learning
algorithms.

"""

import burton_datasets as bd
from hardcoded_classifier import HardCodedClassifier
from burton_knn import BurtonKnnClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def run_test(data, targets, algorithm, test_size = 0.3):
    """
    Splits the data into a training/testing set.
    Builds a model using the provided algorithm.
    Outputs the percent accuracy of the model on the testing set.
    Returns the model that was built.

    :param data: The dataset in a numpy array.
    :param targets: A numpy array of the targets associated with each row in the data.
    :param algorithm: The algorithm to use to build the model. It should
    follow the conventions of a sk-learn algorithm in that it provides a
    .fit() method
    :param test_size: The fraction of rows to include in the test set

    :return: The model
    """
    print("Running Experiment...")
    print("Dataset shape: {}".format(data.shape))

    # Randomizes the order, then breaks the data into training and testing sets
    data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=test_size)

    # Build a model using the provided algorithm
    model = algorithm.fit(data_train, targets_train)

    # Use the model to make a prediction
    targets_predicted = model.predict(data_test)

    # Compute the amount we got correct
    correct = (targets_test == targets_predicted).sum()
    total = len(targets_test)
    percent = correct / total * 100

    # Display result
    print("Correct: {}/{} or {:.2f}%".format(correct, total, percent))

    # Send the model back
    return model


def get_algorithm():
    """
    A factory to create the algorithm we want.
    :return:
    """
    algorithm = GaussianNB()
    #algorithm = HardCodedClassifier()
    #algorithm = KNeighborsClassifier(n_neighbors=3)
    #algorithm = BurtonKnnClassifier(3)

    return algorithm


def get_dataset():
    """
    A factory to load the dataset we want.
    :return:
    """
    data, targets = bd.load_iris()
    #data, targets = bd.load_cars()

    return data, targets


def main():
    """
    Gets and algorithm, dataset, and then passes them to run_tests
    :return:
    """
    algorithm = get_algorithm()
    data, targets = get_dataset()

    model = run_test(data, targets, algorithm)


if __name__ == "__main__":
    main()
