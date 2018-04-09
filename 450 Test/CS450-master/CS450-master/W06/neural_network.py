"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plot

class Neuron():
    #each node (neuron) will have weights and error. nothing else.
    #bias will not have its own node but will have weights and stuff
    #bias is handled differently. 
    def __init__(self, numNodes):
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(0, numNodes)]
        self.error = 0

    #output
    #
    #This function flatten and accounts for bias
    def output(self, results):
        resultsArray = np.append(results, [-1])

        #since the results array is a little funky,
        #we need to do some more modifications to 
        #the results array to make things work
        return self.calculateSigmoid(resultsArray)

    #calculateSigmoid
    #
    #This function simply maps a function to each 
    #element. Then, counts up the sum of the array
    #then calls the sigmoid function to actually
    #squash the number into a number we need
    def calculateSigmoid(self, resultsArray):
        newResults = map(lambda weight, input: weight * input, self.weights, resultsArray)
        
        total = 0

        for (result) in newResults:
            total += result

        return self.sigmoid(total)

    #sigmoid
    #
    # This function squashes the number between 0 and 1
    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

class NeuralNetwork():
    #fit is taking care of everything we need
    def __init__(self):
        pass

    #fit
    #
    #store and prep the data and then call the
    #correct functions to do the job for us
    def fit(self, data, targets, classes):
        self.data = data
        self.targets = targets
        self.classes = classes
        self.numEpochs = -1
        self.numHidden = -1
        self.features = self.data.shape[1]

        self.obtainInformation()
        self.buildNetwork()
        self.train()

        return self

    #train
    #
    #This is the heart of the Neural Network algorithm.
    #TODO: Implement predict algorithm to bring
    #this algorithm to ML algorithm standards with a
    #predict method
    def train(self):
        #store each accuracy results to fulfill
        #one of the requirements
        accuracy = [] 

        #keep track of the final epoch's accuracy so 
        #we don't have to create predict function
        #to help optimize the algorithm.
        finalEpochAccuracy = None

        for _ in range(0, self.numEpochs):
            #keep track of the accuracy for a single epoch
            singleAccuracy = []
            finalEpochAccuracy = 0            
            count = 0

            #zip is actually really nice. it's just like the 
            #cartesian product of two arrays
            #https://docs.python.org/3/library/functions.html#zip
            for (dataPoint, target) in zip(self.data, self.targets):
                results = self.getResults(dataPoint)

                #grab the prediction
                singleAccuracy.append(np.argmax(results[-1]))

                #before we complete the epoch, let's go through and 
                #update the network
                self.updateNetwork(dataPoint, target, results)
            
            #calculating accuracy
            for (index, target) in enumerate(singleAccuracy):
                if (self.targets[index] == target):
                    count += 1

            #storing final epoch accuracy to help optimization
            finalEpochAccuracy = getAccuracy(count, len(self.targets))

            accuracy.append(finalEpochAccuracy)

        #report to the user
        self.plotGraph(accuracy)
        # print("Accuracy: {:.2f}%".format(finalEpochAccuracy))

    #predict
    #
    # It calls predictClass and appends the results of the
    # function call to the predictions array
    def predict(self, data):
        model = []

        for (item) in data:
            model.append(self.predictClass(item))

        return model

    #predictClass
    #
    #gets the results and return the argmax 
    #of that results array
    def predictClass(self, item):
        results = self.getResults(item)

        #return max index
        return np.argmax(results[-1])
    
    #plotGraph
    #
    #This is just a standard plotting function. It takes
    #in an array and plots each element of the array into
    #a graph.
    def plotGraph(self, accuracy):
        plot.plot(accuracy)
        plot.title("Training Accuracy - Neural Network")
        plot.xlabel("# of epochs")
        plot.ylabel("Accuracy")
        plot.show()

    #buildNetwork
    #
    #This function goes through and builds all of the 
    #layers, which then represents a neural network
    def buildNetwork(self):
        #holding all of the layers
        self.layers = []

        #account for bias
        totalSize = self.numHidden + 1

        #a layer needs two things:
        #1) the number of nodes in the previous layer
        #to keep track of weights
        #2) the number of nodes for the layer
        for (index) in range(0, totalSize):
            layer = self.buildLayer(self.featureSize(index), self.getNumNodes(index, self.numHidden))

            self.layers.append(layer)
                                               
    #buildLayer
    #
    #This function goes through and builds a layer
    #of a specified number of nodes
    def buildLayer(self, featuresSize, num):
        nodeLayer = []

        for _ in range(0, num):
            nodeLayer.append(Neuron(featuresSize))

        return nodeLayer

    #featureSize
    #
    #This function simply acts as a getter for the number
    #of nodes in a previous layer.
    def featureSize(self, index):
        #if it is an input layer, there is no previous
        #layer so this checks for that.
        if (index > 0):
            return len(self.layers[index - 1])
        else:
            return self.features

    #getNumNodes
    #
    #This function prompts the user for the number of nodes
    #in a hidden layer. Note that this function is also called for
    #the output layer but the if statement avoids that.
    def getNumNodes(self, index, num):
        #check to see if it is a hidden layer
        if (index < num):
            numNode = -1

            #we can never trust the user...
            try:
                print("How many nodes for Layer {}?".format(index + 1))
                numNode = int(input("> "))

                if (numNode < 1):
                    raise Exception("Please choose more than one node next time.")

            except (ValueError) as err:
                print("ERROR: {}".format(err))

            return numNode
        else:
            return len(self.classes)

    #updateNetwork
    #
    #An epoch has been completed. Let's go through the network
    #and update the neural network.
    def updateNetwork(self, data, target, results):
        #IMPORTANT!
        #we have to calculate the errors before updating the weights
        self.updateErrors(target, results)
        self.updateWeights(data, results)

    #updateErrors
    #
    #This function goes through each node in each layer and calls
    #the function needed to update the error
    def updateErrors(self, target, results):
        for (indexLayer, layer) in reversed(list(enumerate(self.layers))):
            for (indexNode, node) in enumerate(layer):
                node.error = self.getErrorNode(indexNode, indexLayer, target, results)

    #getErrorNode
    #
    #This function goes through the node and calls the correct
    #function to change the error
    def getErrorNode(self, indexNode, indexLayer, target, results):
        nodeResult = None
        hiddenWeight = None
        hiddenError = None

        #here, we know that it is a hidden layer so we prep for a hidden
        #layer error change and call the function
        if (indexLayer < len(results) - 1):
            #get the specific result from the results array to help
            #us change the error
            nodeResult = results[indexLayer][indexNode]

            #call getters to get the info for us
            hiddenWeight = self.getHiddenNodeWeights(indexNode, indexLayer)
            hiddenError = self.getHiddenNodeError(indexLayer)

            return self.getHiddenError(nodeResult, hiddenWeight, hiddenError)
        #here, we know it is an output layer node so we prep data and then
        #call the function do the job for us
        else:
            #get the specific result from the results array to help
            #us change the error  
            nodeResult = results[indexLayer][indexNode]

            #part of the output error change algorith requires
            #(output - target). we create a boolean to help
            #determine if the node matches the target. We
            #can do this since this is an output layer
            #which has access to the targets
            isMatch = indexNode == target

            return self.getOutputError(nodeResult, isMatch)

    #getHiddenNodeError
    #
    #This function acts a getter function to retrieve all
    #of the error for a certain node at a certain layer.
    def getHiddenNodeError(self, indexLayer):
        arrayHiddenError = []

        for (node) in self.layers[indexLayer + 1]:
            arrayHiddenError.append(node.error)
        
        return arrayHiddenError

    #getHiddenNodeWeights
    #
    #This function acts a getter function to retrieve all
    #of the weights for a certain node at a certain layer.
    def getHiddenNodeWeights(self, indexNode, indexLayer):
        arrayHiddenWeights = []

        for (node) in self.layers[indexLayer + 1]:
            arrayHiddenWeights.append(node.weights[indexNode])
        
        return arrayHiddenWeights

    #getHiddenError
    #
    #This function simply calculates the hidden layer error since
    #the output layer error is calculated differently.
    def getHiddenError(self, result, hiddenWeights, hiddenError):
        total = 0

        for (index, weight) in enumerate(hiddenWeights):
            total += (weight * hiddenError[index])

        return result * (1 - result) * total

    #getOutputError
    #
    #This function simply calculates the output layer error since
    #the hidden layer error is calculated differently.
    def getOutputError(self, result, target):
        return result * (1 - result) * (result - target)
        
    #updateWeights
    #
    #This function goes through and updates the weights
    #for each node in each layer. It doesn't matter which
    #direction so we are simply going from input layer to
    #output layer
    def updateWeights(self, data, results):
        for (index, layer) in enumerate(self.layers):
            for (node) in layer:
                #are we at the input layer? if not, we can
                #go ahead and retrieve the previous column's
                #results
                if (index > 0):
                    self.updateNodeWeights(node, results[index - 1])
                #here, we know that we are at the input layer so 
                #convert the input layer into a list so we can account for
                #bias in the function. The results are already in a list
                else:
                    self.updateNodeWeights(node, data.tolist())

    #updateNodeWeights
    #
    #This function goes through and calculates a new
    #array of weights.
    def updateNodeWeights(self, node, data):
        #account for bias
        data = data + [-1]

        newWeightsArray = []

        #iterate through all of the current weights
        for (index, weight) in enumerate(node.weights):
            newWeightsArray.append(weight - (0.1) * data[index] * node.error)

        #store the weights
        node.weights = newWeightsArray

    #getResults
    #
    #This function goes through and calculates
    #the h for each node in each layer and
    #returns the results.
    def getResults(self, input):
        results = []

        for (index, layer) in enumerate(self.layers):
            results.append([node.output(results[index - 1] if index > 0 else input) for node in layer])

        return results

    #obtainInformation
    #
    #This function obtain the numbers of hidden layers and
    #epochs desired. The reason for number of epochs desired
    #is so I can experiment with the number of epochs and how
    #it affects the training of the neural network.
    def obtainInformation(self):
        try:
            print("How many hidden layers? (1, 2)")
            self.numHidden = int(input("> "))

            #it has been proven that if there are more than
            #2 hidden layers in a multi-layer perceptron 
            if (self.numHidden < 1 or self.numHidden > 2):
                raise Exception("please select either 1 or 2 hidden layers.")

            #I really don't want the neural network to overtrain
            #so I am putting a hard cap on the number of epochs
            print("How many epochs? (1 - 10,000)")
            self.numEpochs = int(input("> "))

            #error checking
            if (self.numEpochs < 1 or self.numEpochs > 10000):
                raise Exception("please choose between 1 and 10,000")
        except (ValueError) as err:
            print("ERROR: {}".format(err))

# getAccuracy()
#
# This function calculates and returns the accuracy
def getAccuracy(count, length):
    return (count / length) * 100