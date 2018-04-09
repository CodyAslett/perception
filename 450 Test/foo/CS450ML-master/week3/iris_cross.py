
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn import datasets
from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from knn import kNNClassifier
from sklearn.cross_validation import cross_val_score


# In[17]:


def iris_cross(k , fold):

    iris = datasets.load_iris()
    #shuffle our data

    classifier_1 = kNNClassifier(k)
    custom_scores = cross_val_score(classifier_1, iris.data, iris.target, cv=fold, scoring='accuracy')
    classifier_2 = KNeighborsClassifier(n_neighbors=k)
    sk_scores = cross_val_score(classifier_2, iris.data, iris.target, cv=fold,scoring='accuracy')
    return custom_scores,sk_scores

