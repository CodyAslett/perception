import numpy as np

"""
TODO List
------------
DONE: Allow user to ask for a certain k
DONE: Allow the user to ask for table consisting of k rows

Extra Credit
------------
DONE: Allow the user to choose different datasets

If time
TODO: plot graph

"""

class SamKNNClassifier():
    def __init__(self, k = 3):
        self.k = k

    #we have kNN class that takes care of the core part of the kNN algorithm
    def fit(self, data, target):
        return kNN(data, target, self.k)

class kNN():
    def __init__(self, data, target, k):
        self.data = data
        self.target = target
        self.k = k
        self.model = []

    def predict(self, data):
        #iterate through the test set and call the getClass to classify the test data
        for i in data:
            self.model.append(self.getClass(i, self.k))

        return self.model

    def getClass(self, point, k):
        #create a list of all nearest neightbors
        neighbors = self.nearest(point, k)

        #Credit: https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
        #returns the mode of the list
        return max(neighbors, key = neighbors.count)

    def nearest(self, point, k):
        #Comment for reader:
        #the reason for choosing argsort over sort is to get the class
        #from the target. the machine already knows the training data and target.
        #we want to get the target from the training data to determine the class
        #we are kind of doing clustering stuff in these next few lines

        the_list = []

        #this goes through the entire graph and gets the distance of each neighbors
        dist = np.sum((self.data - point) ** 2, axis = 1)

        #argsort returns the INDEXES of the sorted array.
        dist = np.argsort(dist, axis = 0)

        #left for testing purposes
        # print(dist[:k])
        # print(self.target)

        #iterate through dist list from start to kth spot
        for i in dist[:k]:
            # left for testing purposes
            # print(self.target[i])

            #grabbing the "class" or results
            the_list.append(self.target[i])

        return the_list
