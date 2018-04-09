# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:14:40 2018

@author: Chad
"""
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

iris = datasets.load_iris()
max_iter = 400
param =    {'solver': 'adam', 'learning_rate_init': 0.01}

X = iris.data
Y = iris.target


X = MinMaxScaler().fit_transform(X)

mlp = MLPClassifier(verbose=0, random_state=0, max_iter = max_iter, **param)
mlp.fit(X, Y)

print("Training set score: %f" % mlp.score(X, Y))
#can't use iris because 
df = pd.DataFrame(X)

samp = df.sample(frac=0.3)
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
