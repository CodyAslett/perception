"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""

import kshell
import dtree
import nnShell

def main():
    algorithm = -1

    try:
        print("Please select the number for the following algorithms:")
        print("1 - HardCodedClassifier")
        print("2 - GaussianNB Classifier")
        print("3 - kNN Classifier")
        print("4 - kNN Classifier with incomplete datasets")
        print("5 - Do #4 with k-Fold Cross Validation")
        print("6 - ID3 Decision Tree Classifier")
        print("7 - Multi Layer Neural Network")
        print("8 - Execute sklearn Multi Layer Neural Network")

        algorithm = int(input("> "))

        if (algorithm < 1 or algorithm > 8):
            raise Exception('invalid algorithm selection.')
        elif (algorithm < 6):
            kshell.cs450shell_call(algorithm)
        elif (algorithm == 6):
            dtree.cs450shell(algorithm)
        elif (algorithm == 7 or algorithm == 8):
            nnShell.cs450shell(algorithm)
        else:
            raise Exception('this error should never happen...')
    
    except (ValueError) as err:
        print("ERROR: {}".format(err))

if __name__ == "__main__":
    main()