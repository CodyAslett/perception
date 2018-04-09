"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""

# Import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from hcc import HardCodedClassifier
from knn import SamKNNClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# main()
#
# This function prompts the user for input and initiates the program
def main():
    sklearn_table = -1
    isPrint = -1
    option = -1
    k = -1
    dataset_type = -1

    print("Please select the number for the following algorithms:")
    print("1 - HardCodedClassifier")
    print("2 - GaussianNB")
    print("3 - kNN Classifier")
    #print("4 - Run all algorithms") -- extra credit for next week

    try:
        option = int(input("> "))

        if (option < 0 or option > 3):
            raise Exception('Invalid number!')

        if (option == 3):
            print("Would you like to initiate a battle between SamKNNClassifier() and KNeighborsClassifier() with Iris dataset?")
            print("0 - No")
            print("1 - Yes")
            sklearn_table = int(input("> "))
    except (ValueError) as err:
        print("ERROR: {}".format(err))

    print()

    try:
        if (sklearn_table <= 0):
            print("Would you like to see the test set and target predicted?")
            print("0 - No")
            print("1 - Yes")


            #casting to int to make next if statement to act like a boolean
            isPrint = int(input("> "))

            if (isPrint < 0 or isPrint > 1):
                raise Exception("ERROR: Invalid print_table number!")

        if (sklearn_table <= 0 and option == 3):
            print("What k value would you like to use: ")
            k = int(input("> "))

            if (k < 0):
                raise Exception("ERROR: Invalid k number!")

            print()
            print("Which dataset would you like to use:")
            print("1 - iris")
            print("2 - digits")
            print("3 - wine")
            print("4 - Wisconsin breast cancer")

            dataset_type = int(input("> "))

            if (dataset_type < 1 or dataset_type > 4):
                raise Exception("ERROR: Invalid dataset type number!")

    except (ValueError) as err:
        print("ERROR: {}".format(err))

    #initiate the process
    build_set(option, isPrint, k, sklearn_table, dataset_type)

# build_set()
#
# Purpose: This function splits the dataset into a training and testing set.
def build_set(option, isPrint, k, sklearn_table, dataset_type):
    sk_dataset = None

    # load set depending on user's input
    try:
        if (dataset_type == 1):
            sk_dataset = datasets.load_iris()
        elif (dataset_type == 2):
            sk_dataset = datasets.load_digits()
        elif (dataset_type == 3):
            sk_dataset = datasets.load_wine()
        elif (dataset_type == 4):
            sk_dataset = datasets.load_breast_cancer()
        else:
            raise Exception("ERROR: Invalid number")
    except (ValueError) as err:
        print("ERROR: {}".format(err))

    #splitting up dataset into training set (70%) and testing set (30%)
    train_data, test_data, train_target, test_target = train_test_split(sk_dataset.data, sk_dataset.target, test_size=.3)

    #all of the options have been chosen. let's begin the process.
    run_algorithm(option, isPrint, train_data, test_data, train_target, test_target, k, sklearn_table)

# run_algorithm()
#
# Purpose: This function executes the algorithm and displays the results to the user
def run_algorithm(option, isPrint, train_data, test_data, train_target, test_target, k, sklearn_table):
    count = 0
    classifier = None

    try:
        if option == 1:
            classifier = HardCodedClassifier()
        elif option == 2:
            classifier = GaussianNB()
        elif option == 3:
            classifier = SamKNNClassifier(k)
        else:
            raise Exception('Invalid number!')
    except (ValueError) as err:
        print("ERROR: {}".format(err))

    if (sklearn_table):
        print("SamKNNClassifier() vs KNeighborsClassifier() battle commencing...")
        battle(train_data, test_data, train_target, test_target)
    else:
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

        accuracy = get_accuracy(count, len(test_data))

        print("Accuracy: {:.2f}%".format(accuracy))

# get_accuracy()
#
# This function calculates and returns the accuracy
def get_accuracy(count, length):
    return (count / length) * 100

# battle
#
# This function applies both my kNN and sklearn kNN algorithm and displays
# the results to the users
def battle(train_data, test_data, train_target, test_target):

    print()
    print("#---------------------------------------#")
    print("|-------------- Results ----------------|")
    print("#---------------------------------------#")
    print("|   K    |     Sam      |       SK      |")

    for i in range(1, 10):
        sk_count = 0
        sam_count = 0

        sam_classifier = SamKNNClassifier(i)
        sk_classifier = KNeighborsClassifier(n_neighbors = i)

        model_sam = sam_classifier.fit(train_data, train_target)
        model_sk = sk_classifier.fit(train_data, train_target)

        sam_target_predicted = model_sam.predict(test_data)
        sk_target_predicted = model_sk.predict(test_data)

        for index in range(len(test_data)):
            if sk_target_predicted[index] == test_target[index]:
                sk_count += 1

            if sam_target_predicted[index] == test_target[index]:
                sam_count += 1

        sam_accuracy = get_accuracy(sam_count, len(test_data))
        sk_accuracy = get_accuracy(sk_count, len(test_data))

        print("|   {: ^1}    |    {: ^2}%    |     {: ^1}%    |"\
        .format(i, "{:.2f}".format(sam_accuracy), "{:.2f}".format(sk_accuracy)))

    print("#---------------------------------------#")


if __name__ == "__main__":
    main()
