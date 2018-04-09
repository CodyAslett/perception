import pandas as pd
import numpy as np
from scipy import stats

# get_diabetes
#
# This function goes through and cleans the dataset
def get_diabetes():
    #columns
    headers = [
        'pregnant',
        'glucose',
        'blood pressure',
        'skin fold thickness',
        'serum insulin',
        'bmi',
        'pedigree',
        'age',
        'class'
    ]

    #reads the csv
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
            header=None, names=headers, skipinitialspace=True)

    #replaces the 0.0 with NaN and deletes the rows that consists of NaN
    df.replace(df[headers[0:8]] == 0.0, np.NaN, inplace=True)
    df.dropna(inplace=True)

    #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html
    #convert the dataframe into numpy arrays.
    #the first return is for data while the second return is the targets.
    return stats.zscore(df.as_matrix(columns=headers[0:8])), \
            df.as_matrix(columns=headers[8:9]), \
            ["No", "Yes"]
