
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from knn import kNNClassifier
from sklearn.cross_validation import cross_val_score


# In[60]:


def automobile_mpg(k , fold):

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", delim_whitespace=True, header=None)
    df = df.drop(8, axis=1)
    df = df.replace('?', np.nan)
    df = df.dropna()
    df[3] = pd.to_numeric(df[3])
    #shuffle our data
    df = shuffle(df)
    values = df.values
    X = values[:,1:8]
    y = values[:,0]

    #classifier_1 = kNNClassifier(k)
    #custom_scores = cross_val_score(classifier_1, X, y, cv=fold, scoring='precision')
    #classifier_2 = KNeighborsClassifier(n_neighbors=k)
    #sk_scores = cross_val_score(classifier_2, X, y, cv=fold, scoring='precision')
    #return custom_scores, sk_scores

