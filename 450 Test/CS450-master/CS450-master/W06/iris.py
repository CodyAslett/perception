"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""
import numpy as np
import pandas as pd
from scipy import stats

def get_iris():
    headers = [
        'sepal length',
        'sepal width',
        'petal length',
        'petal width',
        'class'
    ]

    classObj = {
        'Iris-setosa': 1,
        'Iris-versicolor': 2,
        'Iris-virginica': 3
    }

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                    header=None, names=headers)

    df.replace(classObj, inplace=True)

    return stats.zscore(df.as_matrix(columns=headers[:-1])), \
            df.as_matrix(columns=headers[-1:]), \
            ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    