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
from car_washing import car_washing_service
from get_diabetes import get_diabetes
from mpg import rank_mpg
from battle import initiate_battle
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from scipy import stats
import pandas as pd
import numpy as np

# main()
#
# This function prompts the user for input and initiates the program
def main():
    algorithm = -1

    try:
        print("Please select the number for the following algorithms:")
        print("1 - HardCodedClassifier")
        print("2 - GaussianNB Classifier")
        print("3 - kNN Classifier")
        print("4 - kNN Classifier with incomplete datasets")
        print("5 - Do #4 with k-Fold Cross Validation")
        #print("5 - Run all algorithms") -- Extra Credit for later assignments

        algorithm = int(input('> '))

        if (algorithm < 1 or algorithm > 5):
            raise Exception('invalid algorithm selection.')
        elif (algorithm == 5):
            execute_k_fold(algorithm)
        elif (algorithm > 2):
            initiate_knn_algorithm(algorithm)
        else:
            execute_normal(algorithm)

    except (ValueError) as err:
        print('ERROR: {}'.format(err))

# execute_normal
#
# At this point, we know that this is going to be an easy task
# so we prompt the user for a complete dataset and split that
# dataset then call the function to execute the algorithm.
def execute_normal(algorithm):
    train_data, test_data, train_target, test_target = get_dataset(algorithm)

    if (algorithm == 1):
        print('HardCodedClassifier executing now...')

    elif (algorithm == 2):
        print('GaussianNB executing now...')

    execute_algorithm(0, algorithm, train_data, test_data, train_target, test_target, False)

# get_dataset
#
# This function prompts the user for the dataset and it performs the split.
def get_dataset(algorithm):
    dataset_num = prompt_normal_datasets()

    #determine which dataset to load
    if (dataset_num == 1):
        dataset = datasets.load_iris()
    elif (dataset_num == 2):
        dataset = datasets.load_digits()
    elif (dataset_num == 3):
        dataset = datasets.load_wine()
    elif (dataset_num == 4):
        dataset = datasets.load_breast_cancer()

    #splitting up dataset into training set (70%) and testing set (30%)
    return train_test_split(dataset.data, dataset.target, test_size=.3)

# execute_knn
#
# At this point, we know that we will be running some sort of kNN algorithm but with
# a complete dataset. This function prompts the user for a datset then splits it.
# After splitting it, it calls the function to run the algorithm.
def execute_knn(algorithm):
    dataset = None
    dataset_num = prompt_normal_datasets()

    if (dataset_num == 1):
        dataset = datasets.load_iris()
    elif (dataset_num == 2):
        dataset = datasets.load_digits()
    elif (dataset_num == 3):
        dataset = datasets.load_wine()
    elif (dataset_num == 4):
        dataset = datasets.load_breast_cancer()

    return dataset.data, dataset.target

# initiate_knn_algorithm
#
# At this point, we know that the user wants to interact with an kNN algorithm.
# This function takes that and prompts the user to see if they want to see a battle
def initiate_knn_algorithm(algorithm):
    is_battle = -1
    data = None
    target = None
    k = 0
    mpg = None

    try:
        print('Would you like to witness the battle between SamKNNClassifier() and KNeighborsClassifier()?')
        print('Only one will win... Admission is completely free for all those who support SamKNNClassifier()')
        print('0 - No')
        print('1 - Yes')

        is_battle = int(input('> '))

        if (is_battle < 0 or is_battle > 1):
            raise Exception('invalid battle number')
        elif (is_battle == 0):
            print('Whoever is facing into the screen is not an intelligent being.')
            print('Whatever. Let\'s move on.')
            print('About to begin process for normal kNN algorithm execution...')

            k = get_k()

            #determine which datasets to prompt for.
            if (algorithm == 3):
                data, target = execute_knn(algorithm)
            else:
                data, target, mpg = get_dirty_dataset(algorithm)

            #splitting up dataset into training set (70%) and testing set (30%)
            train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=.3)

            #let's execute the algorithm!
            execute_algorithm(k, algorithm, train_data, test_data, train_target, test_target, mpg)

        else:
            print('I must admit that you are one intelligent human.')
            print('SamKNNClassifier() vs KNeighborsClassifier() battle is about to start...')
            print('Entering arena...')

            #determine which prompts of algorithms to prompt the user for.
            if (algorithm == 3):
                data, target = prompt_normal_datasets_for_battle()
            else:
                data, target, mpg = get_dirty_dataset(algorithm)

            #splitting up dataset into training set (70%) and testing set (30%)
            train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=.3)

            #battle starting in 3... 2... 1..
            initiate_battle(train_data, test_data, train_target, test_target)

    except (ValueError) as err:
        print('ERROR: {}'.format(err))

# execute_k_fold
#
# This function performs the 10 fold cross validation.
def execute_k_fold(algorithm):
    sum = 0
    k = get_k()

    train_data, test_data, mpg = get_dirty_dataset(algorithm)

    # 10 is average number of folds
    kf = KFold(n_splits=10)

    #get 10 splits
    kf.get_n_splits(train_data, test_data)

    #looping through and performing the cross validation
    for train_index, test_index in kf.split(train_data, test_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        Y_train, Y_test = test_data[train_index], test_data[test_index]

        #the 10 fold cross validation has been completed. Let's actually execute the algorithm
        sum += execute_algorithm_fold_algorithm(k, 3, X_train, X_test, Y_train, Y_test, mpg)

    if (mpg):
        print("Mean of {} nearest neightbor's MPG: {:.2f} MPG".format(k, sum / 10))
    else:
        #sum is the number that contains all of the result and we are diving it by 10
        #to get the mean - 10 is the number of folds
        print("Accuracy: {:.2f}%".format(sum / 10))

# execute_algorithm_fold_algorithm
#
# This function is used just for the k fold cross validation. This executes the
# algorithm and returns the accuracy or average mpg (if that is the case).
def execute_algorithm_fold_algorithm(k, algorithm, train_data, test_data, train_target, test_target, mpg):
    count = 0
    classifier = None

    try:
        if (algorithm == 1):
            classifier = HardCodedClassifier()
        elif (algorithm == 2):
            classifier = GaussianNB()
        elif (algorithm == 3 or algorithm == 4):
            classifier = SamKNNClassifier(k)
        else:
            raise Exception('Invalid number!')
    except (ValueError) as err:
        print("ERROR: {}".format(err))

    #begin the execution
    model = classifier.fit(train_data, train_target)
    target_predicted = model.predict(test_data)

    if (mpg):
        #here we know that the dataset is involved with MPG which is all regression
        the_mean = np.mean(target_predicted)
        return the_mean
    else:
        #we know here that the dataset is all about classes.
        for index in range(len(test_data)):
            #increment counter for every match from
            #target_predicted and test_target
            if target_predicted[index] == test_target[index]:
                count += 1

        #since we know that this is only part of the k-fold cross Validation
        #we avoid displaying here and return it to be added to the sum.
        return get_accuracy(count, len(test_data))

# execute_algorithm
#
# This function goes through and executes the algorithm. After executing the algorith, it
# presents the score to the user.
def execute_algorithm(k, algorithm, train_data, test_data, train_target, test_target, mpg):
    count = 0
    classifier = None

    #prompt the user if he/she wants to display the target predicted array.
    print_values = prediction_visibility()

    try:
        #create the classifier variable
        if (algorithm == 1):
            classifier = HardCodedClassifier()
        elif (algorithm == 2):
            classifier = GaussianNB()
        elif (algorithm == 3 or algorithm == 4):
            classifier = SamKNNClassifier(k)
        else:
            raise Exception('Invalid number!')
    except (ValueError) as err:
        print("ERROR: {}".format(err))

    model = classifier.fit(train_data, train_target)
    target_predicted = model.predict(test_data)

    #depending on dataset,  the printing of results is different. this handles it
    if (mpg):
        if (print_values):
            for index in range(len(test_data)):
                print('Index: {}. Guess: {}.'.format(index, target_predicted[index]))

        # since this is regression, let's just get the mean and display it
        the_mean = np.mean(target_predicted)
        print("Mean of {} nearest neightbor's MPG: {:.2f} MPG".format(k, the_mean))
    else:
        print()
        for index in range(len(test_data)):
            #increment counter for every match from
            #target_predicted and test_target
            if target_predicted[index] == test_target[index]:
                count += 1

            if print_values:
                print("Index: {} Guess: {} Actual: {}".format(index, target_predicted[index], test_target[index]))
        #print new line
        print()

        #get the accuracy and display it the user

        accuracy = get_accuracy(count, len(test_data))

        print("Accuracy: {:.2f}%".format(accuracy))

# prompt_normal_datasets
#
# This function prompts the user of CLEAN datasets that has no missing values
def prompt_normal_datasets():
    dataset_num = -1

    try:
        print("Which dataset would you like to use:")
        print("1 - iris")
        print("2 - digits")
        print("3 - wine")
        print("4 - Wisconsin breast cancer")

        dataset_num = int(input('> '))

        if (dataset_num < 1 or dataset_num > 4):
            raise Exception('invalid dataset selection.')
        else:
            return dataset_num
    except (ValueError) as err:
        print('ERROR: {}'.format(err))

#prompt_normal_datasets_for_battle
#
#this function is only for the battle and is accommodated for the battle function.
def prompt_normal_datasets_for_battle():
    dataset_num = prompt_normal_datasets()
    dataset = None

    if (dataset_num == 1):
        dataset = datasets.load_iris()
    elif (dataset_num == 2):
        dataset = datasets.load_digits()
    elif (dataset_num == 3):
        dataset = datasets.load_wine()
    elif (dataset_num == 4):
        dataset = datasets.load_breast_cancer()

    return dataset.data, dataset.target

#prediction_visibility
#
#This function prompts the user if he/she wants to see the test set and target predicted.
def prediction_visibility():
    prediction = -1

    try:
        print("Would you like to see the test set and target predicted?")
        print("0 - No")
        print("1 - Yes")

        prediction = int(input('> '))

        #error handling
        if (prediction < 0 or prediction > 1):
            raise Exception('invalid prediction visibility number.')
        else:
            return prediction
    except (ValueError) as err:
        print('ERROR: {}'.format(err))

#get_k
#
#This function prompts the user for k -- only used for kNN algorithms
def get_k():
    try:
        print("What k value would you like to use: ")
        k = int(input("> "))

        if (k < 0):
            raise Exception("ERROR: Invalid k number!")

        return k
    except (ValueError) as err:
        print("ERROR: {}".format(err))

# get_dirty_dataset
#
# This function is to prompt the user for a dirty dataset - which has
# missing values.
def get_dirty_dataset(algorithm):
    data = None
    targets = None
    mpg = False

    print('Launching kNN algorithm with incomplete datasets...')

    print()
    print("Please select a dirty dataset:")
    print("1 - Car")
    print("2 - Pima Indian Diabetes")
    print("3 - Automobile MPG")

    try:
        dirty = int(input("> "))

        if (dirty < 1 or dirty > 3):
            raise Exception('Invalid dataset number')

    except (ValueError) as err:
        print("ERROR: {}".format(err))

    #begin cleansing the filth and scum of the datasets.
    #must not use zscore function on mpg
    if (dirty == 1):
        data, target = car_washing_service()
        stats.zscore(data), stats.zscore(target)
    elif (dirty == 2):
        data, target = get_diabetes()
        stats.zscore(data), stats.zscore(target)
    else:
        data, target = rank_mpg()
        mpg = True

    return data, target, mpg

# get_accuracy()
#
# This function calculates and returns the accuracy
def get_accuracy(count, length):
    return (count / length) * 100

if __name__ == "__main__":
    main()
