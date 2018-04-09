# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 18:37:40 2018

@author: Chad
"""

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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

df2 = readCSV2()
X, Y = separateDataAndTargetsEnd(df2,8)
X = MinMaxScaler().fit_transform(X)
print(X[0,:])
print(Y[0])

max_iter = 800
param =  {'solver': 'adam', 'learning_rate_init': 0.02}  
mlp = MLPClassifier(verbose=0, random_state=0, max_iter = max_iter, **param)
mlp.fit(X, Y)
print("Training set score: %f" % mlp.score(X, Y))

df = pd.DataFrame(X)

samp = df.sample(frac=0.1)
samp2 = samp
samp = samp.values
#print(samp)
#print(samp.index)
print(Y[samp2.index])
#print("SAMP",samp[2])
print(mlp.predict(samp))

smppre = mlp.predict(samp)

count = 0.0
for i in range(len(smppre)):
    if smppre[i] == Y[samp2.index[i]]:
        count += 1
accuracy = count/len(smppre)
print(accuracy)

