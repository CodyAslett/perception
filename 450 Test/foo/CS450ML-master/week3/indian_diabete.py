
# coding: utf-8

# In[63]:


import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from knn import kNNClassifier
from sklearn.cross_validation import cross_val_score


# In[86]:


def indian_diabete(k , fold):

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
                      header=None,skipinitialspace=True)
    df[[1,2,3,4,5]] = df[[1,2,3,4,5]].replace(0, np.NAN)
    df.dropna(inplace=True)
    
    #shuffle our data
    df = shuffle(df)
    values = df.values
    X = values[:,0:8]
    y = values[:,8]

    classifier_1 = kNNClassifier(k)
    custom_scores = cross_val_score(classifier_1, X, y, cv=fold, scoring='accuracy')
    classifier_2 = KNeighborsClassifier(n_neighbors=k)
    sk_scores = cross_val_score(classifier_2, X, y, cv=fold, scoring='accuracy')
    return custom_scores, sk_scores

