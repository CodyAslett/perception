
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from knn import kNNClassifier
from sklearn.cross_validation import cross_val_score


# In[20]:


def carEval(k , fold):
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                      header=None,names=headers, na_values="?", skipinitialspace=True)
    
    #shuffle our data
    df = shuffle(df)

    cleanup_nums = {"doors":    {"5more": 5},
                "persons":  {"more": 5},
                "lug_boot": {"small": 1, "med": 2, "big": 3},
                "safety":   {"low": 1, "med": 2, "high": 3},
                "buying":   {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                "maint":    {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                "class":    {"unacc": 1, "acc": 2, "good": 3, "vgood": 4}}
    df.replace(cleanup_nums, inplace=True)
    df[['doors','persons']] = df[['doors','persons']].apply(pd.to_numeric)
    #here is anything but the class
    X = np.array(df.iloc[:,0:6])
    #the class target
    y = np.array(df['class']) 
    classifier_1 = kNNClassifier(k)
    custom_scores = cross_val_score(classifier_1, X, y, cv=fold, scoring='accuracy')
    classifier_2 = KNeighborsClassifier(n_neighbors=k)
    sk_scores = cross_val_score(classifier_2, X, y, cv=fold, scoring='accuracy')
    return custom_scores, sk_scores

