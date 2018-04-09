"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""

# Import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from hcc import HardCodedClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

# main()
#
# This function prompts the user for input and initiates the program
def main():
    print("Please select the number for the following algorithms:")
    print("1 - HardCodedClassifier")
    print("2 - GaussianNB")

    try:
        option = int(input("> "))

        if (option < 0 or option > 2):
            raise Exception('Invalid number!')
    except (ValueError) as err:
        print("ERROR: {}".format(err))

    print()
    print("Would you like to see the test set and target predicted?")
    print("0 - No")
    print("1 - Yes")

    try:
        #casting to int to make next if statement to act like a boolean
        isPrint = int(input("> "))

        if (isPrint < 0 or isPrint > 1):
            raise Exception('Invalid number!')
    except (ValueError) as err:
        print("ERROR: {}".format(err))

    #initiate the process
    build_set(option, isPrint)

# build_set()
#
# Purpose: This function splits the dataset into a training and testing set.
def build_set(option, isPrint):
    # load set
    iris = datasets.load_iris()

    #splitting up dataset into training set (70%) and testing set (30%)
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=.3)

    #all of the options have been chosen. let's begin the process.
    run_algorithm(option, isPrint, train_data, test_data, train_target, test_target)

# run_algorithm()
#
# Purpose: This function executes the algorithm and displays the results to the user
def run_algorithm(option, isPrint, train_data, test_data, train_target, test_target):
    count = 0
    classifier = None

    try:
        if option == 1:
            classifier = HardCodedClassifier()
        elif option == 2:
            classifier = GaussianNB()
        else:
            raise Exception('Invalid number!')
    except (ValueError) as err:
        print("ERROR: {}".format(err))

    #.fit() trains the machine
    model = classifier.fit(train_data, train_target)

    target_predicted = model.predict(test_data)

    print()
    for index in range(len(test_data)):
        #increment counter for every match from
        #target_predicted and test_target
        if target_predicted[index] == test_target[index]:
            count += 1

        if isPrint:
            print("Index: {} Guess: {} Actual: {}".format(index, target_predicted[index], test_target[index]))
    #print new line
    print()

    accuracy = (count / len(test_data)) * 100

    print("Accuracy: {:.2f}%".format(accuracy))


if __name__ == "__main__":
    main()
