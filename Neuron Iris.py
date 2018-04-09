# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:52:04 2018

@author: Chad
"""

import sys
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
def readCSV2():    
    filename = './pima-indians-diabetes.data'
    data = pd.read_csv(filename, skipinitialspace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['pregnant', 'plasma_glucose_level','Diastolic blood pressure', 'skin fold', 'serum insulin', 'BMI', 'Diabetes pedigree function', 'age(years)','targets1']
    return df

def separateDataAndTargetsEnd(dataframe,a):
    target = dataframe.iloc[:,a]
    target = pd.DataFrame.as_matrix(target,columns=None)
    dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=True)
    dat = pd.DataFrame.as_matrix(dataframe,columns=None)
    return dat, target

def experimentalShell(data_train, data_test , targets_train, targets_test, classifier):
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    return targets_predicted



class perception:
    accuricy = 0
    def __init__(self, forLayerParameter, numInputs, data, targets, df2):
            
        df2 = df2[df2.iloc[:,2] != 0]
        
        data = preprocessing.normalize(data, norm='l2')  
        
        #number of inputs
        
        Classi = NNClassifier()
        model = Classi.fit(data, targets)
        targets = targets
    
        layerList = model.createLayer(forLayerParameter, numInputs)
        array = []
        for i in range(100):
            listtry = model.feedForward(layerList, data[i])
            layerList,predictons = model.backwardProp(listtry, targets[i])
            if predictons[1] > predictons[0]:
                array.append(1) 
            else:
                array.append(0)
    
        count = 0.
        for i in range(len(array)):
            if int(array[i]) == int(targets[i]):
                count += 1
        self.accuracy = count/len(array)
        return 
        
    def get(self):
        return self.accuracy

    #####################################################################
 # Neural Network Model
 # 
#####################################################################   
class NNModel:
    def __init__(self, data_train, targets_train):
        self.data = data_train
        self.targets = targets_train
        return None
    
    def addBias(self,inputs):
        return np.append(inputs, -1)
    
    def checkActivation(self,inputs, weights):
        return np.dot(inputs,weights)
    
    def getActivation(self, hval):
        return 1/(1 + np.exp(-hval))
    
    def makeNode(self, size):
        weights = []
        activationVal = 0
        for i in range(size+1): 
            weights.append(random.uniform(-.5, .5))
        return weights, activationVal
    
    def createNodelist(self, numNodes, numWeights):
        nodeList = []
        for i in range(numNodes): 
            nodeList.append(self.makeNode(numWeights))
        return nodeList
    
    def createLayer(self, layerArray, numInputs):
        layerList = []
        layerList.append(self.createNodelist(layerArray[0],numInputs))
        
        for i in range(len(layerArray)-1):
            layerList.append(self.createNodelist(layerArray[i+1],layerArray[i]))
                    
  
        return layerList
    
    def feedForward(self, layerList, inputs):
        inputAndBias = self.addBias(inputs)
        array = []
        array.append(-1)
        for i in range(len(layerList[0])):
            hval = self.checkActivation(inputAndBias,np.transpose(layerList[0][i][0]))
            activationVal = self.getActivation(hval)
            layerList[0][i]=  layerList[0][i][0],activationVal
            array.append(layerList[0][i][1])
        
        activationArray = [[] for d in range(len(layerList))]
        weightsArray = [[] for d in range(len(layerList))]
        for i in range(len(layerList)):
            for j in range(len(layerList[i])):
                weightsArray[i].append(layerList[i][j][0])
                activationArray[i].append(layerList[i][j][1])
        activationArray[0].append(-1)    
            
        for i in range(len(layerList)-1):
            if i > 0:
                activationArray[i].append(-1)        
        for i in range(len(layerList)-1):       
            for j in range(len(layerList[i+1])):
                hval = self.checkActivation(activationArray[i],np.transpose(weightsArray[i+1][j]))
                activationVal = self.getActivation(hval)
                activationArray[i+1][j] = activationVal
        for i in range(len(layerList)):       
            for j in range(len(layerList[i])):  
                layerList[i][j] = weightsArray[i][j],activationArray[i][j]
        return layerList
#        
    def backwardProp(self, layerList, target):
        errorArray = [[] for d in range(len(layerList))]
        for j in range(len(layerList[-1])):
            errorArray[-1].append(layerList[-1][j][1]*(1-layerList[-1][j][1])*(layerList[-1][j][1]  - target))
        predictions = []
        for i in range(len(layerList[-1])):
            predictions.append(layerList[-1][i][1])
        for i in range(len(layerList)-1):
            for j in range(len(layerList[-(i+2)])):
                errorArray[-(i+2)].append(0) 
        for i in range(len(layerList)-1):
            for j in range(len(layerList[-(i+2)])):
                for k in range(len(layerList[-(i+1)])):
                    errorArray[-(i+2)][j] += layerList[-(i+2)][j][1]*(1-layerList[-(i+2)][j][1])*(errorArray[-(i+1)][k] * layerList[-(i+1)][k][0][j])
        lr = .2
        for i in range(len(layerList)):
            for j in range(len(layerList[i])):
                for k in range(len(layerList[i][j][0])):
                    layerList[i][j][0][k] = layerList[i][j][0][k] - lr*errorArray[i][j]*layerList[i][j][1]        
        for i in range(len(layerList)):
            for j in range(len(layerList[i])):
                layerList[i][j] = layerList[i][j][0],0 
        self.saveError(errorArray[-1][0], errorArray[-1][1], errorArray[-1][2])
        
        return layerList,predictions
    
    def saveError(self, valOne, valTwo, valThree):
        file = open('report2.csv', 'a')
        file.write('%f, %f, %f,\n' % (valOne, valTwo, valThree))
        file.close()
        
        
class NNClassifier:
    def __init__(self):
        pass
        return
    
    def fit(self, data, targets):
        m = NNModel(data, targets)
        return m


def main(argv):
    avg = 0
    total = 0
    count = 0
    df1 = pd.DataFrame(datasets.load_iris().data)
    df2 = pd.DataFrame(datasets.load_iris().target)
    df_c = pd.concat([df1, df2], axis=1)
    print(df_c)
    
    for i in range(1) :
        df2 = df_c
        data, targets = separateDataAndTargetsEnd(df2,4)
        data = MinMaxScaler().fit_transform(data)
        numInputs = len(data[0])

        forLayerParameter = [3,2,6,3]
        nuralNetwork = perception(forLayerParameter, numInputs, data, targets, df2)
        acc = nuralNetwork.get()
        total += acc
        count += 1
        print(count, " Accuracy =, ", acc)       
    avg = total/count
    print("avg = ", avg)

    


if __name__== "__main__":
    main(sys.argv)
    
    
